"""In-silico randomised controlled trials and power analysis.

Workflow
--------
1. :func:`build_trial_pool` screens a virtual population, applies the
   eligibility criteria of a :class:`TrialDesign` at each patient's
   enrolment visit and simulates every eligible patient twice, untreated
   and treated from enrolment. Because both *potential outcomes* are known,
   individual treatment effects are available too
   (:meth:`TrialPool.individual_effects`).
2. :func:`run_trials` repeatedly draws two arms from the pool, adds
   dropout and measurement noise, and analyses each replicate trial on
   several endpoints. The fraction of significant replicates estimates
   power (or type-I error, for a placebo).

The analyses are deliberately simple and transparent (per-patient OLS
slopes with Welch's t-test, log-rank, a CAFS-style joint rank). They are
meant for comparing designs, not for replicating a specific statistical
analysis plan.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .cohort import DEFAULT_POPULATION, PopulationModel
from .config import DEFAULT_CONFIG, PhysiologyConfig
from .engine import healthy_nfl, simulate
from .metrics import slopes
from .patient import Patient
from .regions import region_index
from .result import SimulationResult
from .scoring import observe_items
from .treatment import Treatment, TreatmentLike, as_treatment_list, treatment_label

ENDPOINTS: tuple[str, ...] = ("alsfrs_slope", "survival", "cafs", "mune_slope", "fvc_slope", "nfl")
"""Endpoints analysed in every replicate trial."""


@dataclass(frozen=True)
class TrialDesign:
    """Design of a two-arm, parallel-group trial (1:1 randomisation).

    Parameters
    ----------
    n_per_arm:
        Patients randomised to each arm.
    duration:
        Follow-up from randomisation (months).
    visit_interval:
        Months between assessments.
    max_months_since_onset:
        Eligibility: symptom duration at enrolment.
    min_fvc:
        Eligibility: respiratory capacity (% predicted) at enrolment.
    min_delta_fs:
        Eligibility: progression rate since onset, (48 - score)/months, to
        enrich for faster progressors (0 = no enrichment).
    genotype:
        Enrol only this genotype (``None`` = all).
    enrollment_window:
        Enrolment happens uniformly within this many months after diagnosis.
    dropout_rate:
        Exponential rate (per month) of withdrawal for reasons unrelated to
        the disease.
    item_noise_sd:
        Rater noise on each functional item before rounding.
    mune_noise_cv:
        Multiplicative (log-normal) noise of motor unit number estimates.
    mune_region:
        Region whose motor-unit pool is followed electrophysiologically, or
        ``"onset"`` for each patient's region of onset (the usual choice of a
        clinically affected muscle).
    fvc_noise_sd:
        Additive noise on FVC (% predicted points).
    nfl_noise_cv:
        Visit-to-visit (log-normal) variability of plasma NfL, assay included.
    min_nfl_fold:
        Eligibility: baseline NfL as a multiple of the healthy level for age
        (enrichment for active degeneration; 0 = none).
    alpha:
        Two-sided significance level.
    """

    n_per_arm: int = 150
    duration: float = 12.0
    visit_interval: float = 1.0
    max_months_since_onset: float = 24.0
    min_fvc: float = 70.0
    min_delta_fs: float = 0.0
    min_nfl_fold: float = 0.0
    genotype: str | None = None
    enrollment_window: float = 3.0
    dropout_rate: float = 0.01
    item_noise_sd: float = 0.3
    mune_noise_cv: float = 0.15
    mune_region: str = "onset"
    fvc_noise_sd: float = 5.0
    nfl_noise_cv: float = 0.2
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.n_per_arm < 2:
            raise ValueError("n_per_arm must be >= 2")
        if self.duration <= 0 or self.visit_interval <= 0:
            raise ValueError("duration and visit_interval must be > 0")
        if self.visit_interval > self.duration:
            raise ValueError("visit_interval must not exceed duration")
        if self.mune_region != "onset":
            region_index(self.mune_region)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

    def replace(self, **changes: Any) -> TrialDesign:
        return replace(self, **changes)

    @property
    def visit_times(self) -> np.ndarray:
        n = math.floor(self.duration / self.visit_interval + 1e-9)
        return np.arange(n + 1) * self.visit_interval


@dataclass(frozen=True, eq=False)
class TrialPool:
    """Eligible virtual patients with both potential outcomes.

    Arrays with a leading axis of 2 hold ``[control, treated]`` outcomes.
    Times on the visit axis are months since enrolment.
    """

    design: TrialDesign
    treatment: tuple[Treatment, ...]
    patients: tuple[Patient, ...]
    enrollment: np.ndarray
    months_since_onset: np.ndarray
    baseline_alsfrs: np.ndarray
    baseline_delta_fs: np.ndarray
    visit_times: np.ndarray
    items: np.ndarray  # (2, n, K, n_items)
    fvc: np.ndarray  # (2, n, K)
    nfl: np.ndarray  # (2, n, K)
    mune: np.ndarray  # (2, n, K)
    event: np.ndarray  # (2, n) months from enrolment; inf if beyond horizon
    n_screened: int

    @property
    def size(self) -> int:
        return len(self.patients)

    @property
    def screen_failure_rate(self) -> float:
        return 1.0 - self.size / self.n_screened

    def __repr__(self) -> str:
        return (
            f"TrialPool({self.size} eligible of {self.n_screened} screened, "
            f"treatment: {treatment_label(list(self.treatment))})"
        )

    def individual_effects(self) -> pd.DataFrame:
        """Per-patient treatment effects (treated minus control potential outcome).

        Outcomes are only simulated up to the end of follow-up, so event times
        are NaN when no event occurs during follow-up; ``event_delay`` is
        defined only when both arms have the event, and ``event_prevented``
        flags events avoided during follow-up.
        """
        alsfrs = self.items.sum(-1)
        end = alsfrs[:, :, -1]
        within = self.event <= self.design.duration
        ev_c = np.where(within[0], self.event[0], np.nan)
        ev_t = np.where(within[1], self.event[1], np.nan)
        return pd.DataFrame(
            {
                "patient": [p.name for p in self.patients],
                "onset_region": [p.onset_region for p in self.patients],
                "genotype": [p.genotype for p in self.patients],
                "baseline_alsfrs": self.baseline_alsfrs,
                "baseline_delta_fs": self.baseline_delta_fs,
                "alsfrs_end_control": end[0],
                "alsfrs_end_treated": end[1],
                "alsfrs_preserved": end[1] - end[0],
                "event_control": ev_c,
                "event_treated": ev_t,
                "event_delay": ev_t - ev_c,
                "event_prevented": within[0] & ~within[1],
            }
        )


def _sample_screening(
    population: PopulationModel, n: int, rng: np.random.Generator, offset: int
) -> list[Patient]:
    seed = int(rng.integers(0, 2**31 - 1))
    return population.sample(n, seed=seed, name_prefix=f"s{offset:03d}_")


def _gather(series: np.ndarray, t: np.ndarray, record_dt: float) -> np.ndarray:
    """Interpolate (P, T, ...) recorded series at per-patient times t (P, K)."""
    n_t = series.shape[1]
    pos = np.clip(t / record_dt, 0.0, n_t - 1.0)
    i0 = np.minimum(pos.astype(np.intp), n_t - 2)
    w = pos - i0
    rows = np.arange(series.shape[0])[:, None]
    a = series[rows, i0]
    b = series[rows, i0 + 1]
    w = w.reshape(w.shape + (1,) * (series.ndim - 2))
    return a + w * (b - a)


def build_trial_pool(
    treatment: TreatmentLike,
    design: TrialDesign = TrialDesign(),
    *,
    population: PopulationModel = DEFAULT_POPULATION,
    pool_size: int | None = None,
    seed: int | None = 0,
    dt: float = 0.5,
    horizon: float = 132.0,
    max_screened: int = 200_000,
    config: PhysiologyConfig = DEFAULT_CONFIG,
) -> TrialPool:
    """Screen a virtual population and simulate eligible patients under both arms.

    Parameters
    ----------
    treatment:
        Intervention given to the active arm from enrolment (its ``start`` is
        measured from each patient's enrolment).
    design:
        Trial design (eligibility, duration, visits).
    population:
        Population the screened patients come from.
    pool_size:
        Number of eligible patients to collect; defaults to
        ``max(2000, 10 * n_per_arm)``. Replicate trials resample this pool,
        so power estimates are *conditional on the pool*. Rare endpoints
        (e.g. deaths in a short trial) are most sensitive to this; compare a
        few seeds, or enlarge the pool, before relying on small differences.
    seed:
        Random seed (screening population, enrolment times).
    dt:
        Engine time step (months).
    horizon:
        Simulated months since disease onset. Patients whose follow-up would
        extend beyond it are not enrolled.
    """
    treatments = as_treatment_list(treatment)
    rng = np.random.default_rng(seed)
    if pool_size is None:
        pool_size = max(2000, 10 * design.n_per_arm)
    visits = design.visit_times
    record_dt = dt
    chosen: list[Patient] = []
    enroll: list[float] = []
    n_screened = n_simulated = n_eligible_seen = 0
    batch = max(256, int(pool_size * 1.7))
    control_parts: list[SimulationResult] = []
    control_idx: list[np.ndarray] = []

    while len(chosen) < pool_size:
        if n_screened >= max_screened:
            raise RuntimeError(
                f"only {len(chosen)} eligible patients after screening {n_screened}; "
                "relax the eligibility criteria or raise max_screened"
            )
        screen = _sample_screening(population, batch, rng, len(control_parts))
        delay = np.array([p.diagnostic_delay for p in screen])
        offset = delay + rng.uniform(0.0, design.enrollment_window, batch)
        # Simulate patients of similar speed together so whole chunks can stop
        # early, then restore the random screening order before selecting, so
        # a partially used batch does not favour fast progressors.
        order = np.argsort([-p.lmn_rate for p in screen], kind="stable")
        res = simulate(
            [screen[i] for i in order],
            None,
            months=horizon,
            dt=dt,
            record_every=record_dt,
            config=config,
            stop_after_onset=(offset + design.duration)[order],
        ).subset(np.argsort(order))
        onset = res.symptom_onset
        t_enroll = onset + offset
        with np.errstate(invalid="ignore"):
            alive = ~(res.event_time <= t_enroll)
            fits = t_enroll + design.duration <= horizon
            duration_ok = (t_enroll - onset) <= design.max_months_since_onset
        fvc0 = _gather(res.fvc, np.nan_to_num(t_enroll)[:, None], record_dt)[:, 0]
        nfl0 = _gather(res.nfl, np.nan_to_num(t_enroll)[:, None], record_dt)[:, 0]
        nfl_fold0 = nfl0 / healthy_nfl(np.array([p.age_at_onset for p in screen]), config)
        score0 = _gather(res.alsfrs, np.nan_to_num(t_enroll)[:, None], record_dt)[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            dfs = (res.max_score - score0) / (t_enroll - onset)
        ok = (
            np.isfinite(onset)
            & alive
            & fits
            & duration_ok
            & (fvc0 >= design.min_fvc)
            & (dfs >= design.min_delta_fs)
            & (nfl_fold0 >= design.min_nfl_fold)
        )
        if design.genotype is not None:
            ok &= np.array([p.genotype == design.genotype for p in screen])
        idx = np.flatnonzero(ok)[: pool_size - len(chosen)]
        # Count only patients screened up to the last one enrolled.
        n_screened += batch if len(chosen) + idx.size < pool_size else int(idx[-1]) + 1
        n_eligible_seen += int(ok.sum())
        n_simulated += batch
        if idx.size:
            chosen.extend(screen[i] for i in idx)
            enroll.extend(t_enroll[idx])
            control_parts.append(res)
            control_idx.append(idx)
        # Size the next batch from the eligibility rate observed so far.
        rate = max(n_eligible_seen / n_simulated, 0.02)
        batch = int(min(max(64, 1.15 * (pool_size - len(chosen)) / rate + 32), max_screened))

    enrollment = np.array(enroll)
    patients = tuple(chosen)
    treated = simulate(
        patients,
        treatments,
        months=horizon,
        dt=dt,
        record_every=record_dt,
        config=config,
        time_origin=enrollment,
        stop_time=enrollment + design.duration,
    )
    t_visits = enrollment[:, None] + visits[None, :]
    if design.mune_region == "onset":
        mune_r = np.array([region_index(p.onset_region) for p in patients])
    else:
        mune_r = np.full(len(patients), region_index(design.mune_region))

    def outcomes(results: Sequence[tuple[SimulationResult, np.ndarray]]):
        items, fvc, mune, event, nfl = [], [], [], [], []
        start = 0
        for res, idx in results:
            tv = t_visits[start : start + idx.size]
            items.append(_gather(res.items[idx], tv, record_dt))
            fvc.append(_gather(res.fvc[idx], tv, record_dt))
            r = mune_r[start : start + idx.size]
            mune.append(_gather(res.mu_count[idx][np.arange(idx.size), :, r], tv, record_dt))
            event.append(res.event_time[idx])
            nfl.append(_gather(res.nfl[idx], tv, record_dt))
            start += idx.size
        return (
            np.concatenate(items),
            np.concatenate(fvc),
            np.concatenate(mune),
            np.concatenate(event),
            np.concatenate(nfl),
        )

    c_items, c_fvc, c_mune, c_event, c_nfl = outcomes(list(zip(control_parts, control_idx, strict=True)))
    t_items, t_fvc, t_mune, t_event, t_nfl = outcomes([(treated, np.arange(len(patients)))])

    events = np.stack([c_event, t_event]) - enrollment
    events = np.where(np.isfinite(events), events, np.inf)

    sym = np.concatenate(
        [res.symptom_onset[idx] for res, idx in zip(control_parts, control_idx, strict=True)]
    )
    base_score = c_items[:, 0].sum(-1)
    return TrialPool(
        design=design,
        treatment=tuple(treatments),
        patients=patients,
        enrollment=enrollment,
        months_since_onset=enrollment - sym,
        baseline_alsfrs=base_score,
        baseline_delta_fs=(c_items.shape[-1] * 4 - base_score) / (enrollment - sym),
        visit_times=visits,
        items=np.stack([c_items, t_items]),
        fvc=np.stack([c_fvc, t_fvc]),
        nfl=np.stack([c_nfl, t_nfl]),
        mune=np.stack([c_mune, t_mune]),
        event=events,
        n_screened=n_screened,
    )


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def welch_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (mean(x) - mean(y), two-sided Welch p-value), ignoring NaNs."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    diff = float(x.mean() - y.mean())
    se = math.sqrt(vx + vy)
    if se == 0:
        return diff, 1.0 if diff == 0 else 0.0
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1))
    return diff, float(2.0 * stats.t.sf(abs(diff / se), df))


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    """Log-rank test for two groups.

    Returns ``(hazard ratio of group 1 vs group 0, two-sided p)``; the hazard
    ratio is Peto's one-step estimate ``exp((O - E) / V)``.
    """
    order = np.lexsort((~event.astype(bool), time))
    e = event[order].astype(float)
    g = group[order].astype(float)
    n = time.size
    at_risk = n - np.arange(n, dtype=float)
    at_risk_1 = np.cumsum(g[::-1])[::-1]
    frac = at_risk_1 / at_risk
    o_minus_e = float(np.sum(e * (g - frac)))
    var = float(np.sum(e * frac * (1.0 - frac)))
    if var <= 0:
        return float("nan"), float("nan")
    z = o_minus_e / math.sqrt(var)
    return float(math.exp(o_minus_e / var)), float(2.0 * stats.norm.sf(abs(z)))


def cafs_scores(time: np.ndarray, died: np.ndarray, function: np.ndarray) -> np.ndarray:
    """Combined Assessment of Function and Survival (Berry et al., 2013) scores.

    Every patient is compared with every other: an observed earlier death
    loses; otherwise the patient with the better functional outcome (here,
    the less negative slope) wins. Returns the net number of wins per patient.
    """
    t_i, t_j = time[:, None], time[None, :]
    d_i, d_j = died[:, None].astype(bool), died[None, :].astype(bool)
    i_better_surv = d_j & (t_i > t_j)
    j_better_surv = d_i & (t_j > t_i)
    decided = i_better_surv | j_better_surv
    f_i, f_j = function[:, None], function[None, :]
    with np.errstate(invalid="ignore"):
        func = np.where(np.isfinite(f_i) & np.isfinite(f_j), np.sign(f_i - f_j), 0.0)
    surv = i_better_surv.astype(float) - j_better_surv.astype(float)
    score = np.where(decided, surv, func)
    return score.sum(axis=1)


def _analyse(
    rng: np.random.Generator,
    pool: TrialPool,
    design: TrialDesign,
    idx: np.ndarray,
    arm: np.ndarray,
) -> dict[str, float]:
    keep = pool.visit_times <= design.duration + 1e-9
    visits = pool.visit_times[keep]
    items = pool.items[arm, idx][:, keep]  # (n, K, I)
    event = pool.event[arm, idx]
    fvc = pool.fvc[arm, idx][:, keep]
    mune = pool.mune[arm, idx][:, keep]
    nfl = pool.nfl[arm, idx][:, keep]

    if design.dropout_rate > 0:
        dropout = rng.exponential(1.0 / design.dropout_rate, idx.size)
    else:
        dropout = np.full(idx.size, np.inf)
    censor = np.minimum(dropout, design.duration)
    died = event <= censor
    follow = np.where(died, event, censor)
    seen = visits[None, :] <= follow[:, None] + 1e-9
    seen &= ~(died[:, None] & (visits[None, :] >= event[:, None]))

    score = observe_items(items, rng, design.item_noise_sd).sum(-1)
    score = np.where(seen, score, np.nan)
    noisy_mune = mune * np.exp(rng.normal(0.0, design.mune_noise_cv, mune.shape))
    log_mune = np.where(seen, np.log(np.maximum(noisy_mune, 0.5)), np.nan)
    noisy_fvc = np.where(seen, fvc + rng.normal(0.0, design.fvc_noise_sd, fvc.shape), np.nan)

    tx = arm == 1
    out: dict[str, float] = {}
    s_alsfrs = slopes(visits, score)
    eff, p = welch_test(s_alsfrs[tx], s_alsfrs[~tx])
    out.update(alsfrs_slope_effect=eff, alsfrs_slope_p=p)
    ctl = s_alsfrs[~tx]
    ctl = ctl[np.isfinite(ctl)]
    ctl_mean = float(ctl.mean()) if ctl.size else float("nan")
    out["alsfrs_slope_slowing"] = -eff / ctl_mean if ctl_mean else float("nan")

    hr, p = logrank_test(follow, died, tx)
    out.update(survival_effect=hr, survival_p=p)
    out["events"] = float(died.sum())

    cafs = cafs_scores(follow, died, s_alsfrs)
    u = stats.mannwhitneyu(cafs[tx], cafs[~tx], alternative="two-sided")
    out.update(cafs_effect=float(cafs[tx].mean() - cafs[~tx].mean()) / idx.size, cafs_p=float(u.pvalue))

    s_mune = slopes(visits, log_mune)
    eff, p = welch_test(s_mune[tx], s_mune[~tx])
    out.update(mune_slope_effect=eff, mune_slope_p=p)

    s_fvc = slopes(visits, noisy_fvc)
    eff, p = welch_test(s_fvc[tx], s_fvc[~tx])
    out.update(fvc_slope_effect=eff, fvc_slope_p=p)

    # NfL: change in log NfL from baseline to the last observed visit, reported
    # as the treated/control geometric mean ratio (as in gene-therapy trials).
    log_nfl = np.log(nfl * np.exp(rng.normal(0.0, design.nfl_noise_cv, nfl.shape)))
    log_nfl = np.where(seen, log_nfl, np.nan)
    has_follow_up = np.isfinite(log_nfl[:, 1:]).any(axis=1)
    last = np.array([row[np.isfinite(row)][-1] for row in log_nfl])
    change = np.where(has_follow_up, last - log_nfl[:, 0], np.nan)
    eff, p = welch_test(change[tx], change[~tx])
    out.update(nfl_effect=float(np.exp(eff)), nfl_p=p)
    return out


#: Direction of benefit for each endpoint effect (+1: larger is better).
_BENEFIT = {"alsfrs_slope": 1, "survival": -1, "cafs": 1, "mune_slope": 1, "fvc_slope": 1, "nfl": -1}
#: Effects reported as ratios (null value 1) rather than differences (null 0).
_RATIO_ENDPOINTS = ("survival", "nfl")


@dataclass(frozen=True, eq=False)
class TrialResults:
    """Replicate-trial outcomes; ``table`` has one row per replicate."""

    design: TrialDesign
    treatment_name: str
    table: pd.DataFrame

    def power(self) -> pd.DataFrame:
        """Fraction of replicates significant *in favour of* treatment, per endpoint.

        For a placebo this is the one-sided false-positive rate (about alpha/2).
        """
        rows = []
        for ep in ENDPOINTS:
            p = self.table[f"{ep}_p"]
            eff = self.table[f"{ep}_effect"]
            favour = np.sign(eff - (1.0 if ep in _RATIO_ENDPOINTS else 0.0)) == _BENEFIT[ep]
            sig = (p < self.design.alpha) & favour
            n = int(p.notna().sum())
            power = float(sig.sum() / n) if n else float("nan")
            se = math.sqrt(power * (1 - power) / n) if n else float("nan")
            rows.append(
                {
                    "endpoint": ep,
                    "power": power,
                    "power_se": se,
                    "median_effect": float(eff.median()),
                    "any_significant": float(((p < self.design.alpha).sum()) / n) if n else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"TrialResults({len(self.table)} replicates, treatment: {self.treatment_name})"


def run_trials(
    pool: TrialPool,
    n_trials: int = 500,
    *,
    n_per_arm: int | None = None,
    duration: float | None = None,
    seed: int | None = 0,
) -> TrialResults:
    """Simulate replicate trials by resampling arms from a :class:`TrialPool`.

    ``n_per_arm`` and ``duration`` (shorter than the pool's follow-up)
    override the pool's design, so one pool serves many designs.
    """
    design = pool.design
    if n_per_arm is not None:
        design = design.replace(n_per_arm=n_per_arm)
    if duration is not None:
        if duration > pool.design.duration + 1e-9:
            raise ValueError("duration cannot exceed the pool's follow-up")
        design = design.replace(duration=duration, visit_interval=min(design.visit_interval, duration))
    n = design.n_per_arm
    if 2 * n > pool.size:
        raise ValueError(f"pool of {pool.size} is too small for 2 x {n} patients")
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_trials):
        idx = rng.choice(pool.size, size=2 * n, replace=False)
        arm = np.repeat([1, 0], n)
        row = _analyse(rng, pool, design, idx, arm)
        row["replicate"] = k
        rows.append(row)
    table = pd.DataFrame(rows).set_index("replicate")
    return TrialResults(design, treatment_label(list(pool.treatment)), table)


def power_curve(
    pool: TrialPool,
    sample_sizes: Sequence[int],
    n_trials: int = 300,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Power of every endpoint across sample sizes (per arm), from one pool."""
    frames = []
    for n in sample_sizes:
        res = run_trials(pool, n_trials, n_per_arm=int(n), seed=seed)
        df = res.power()
        df.insert(0, "n_per_arm", int(n))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def power_by_duration(
    pool: TrialPool,
    durations: Sequence[float],
    n_per_arm: int | None = None,
    n_trials: int = 300,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Power of every endpoint for trials stopped after each follow-up duration."""
    frames = []
    for d in durations:
        res = run_trials(pool, n_trials, n_per_arm=n_per_arm, duration=float(d), seed=seed)
        df = res.power()
        df.insert(0, "duration", float(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate_trial(
    treatment: TreatmentLike,
    design: TrialDesign = TrialDesign(),
    *,
    n_trials: int = 500,
    seed: int | None = 0,
    **pool_kwargs: Any,
) -> TrialResults:
    """One-call convenience: build a pool and run replicate trials."""
    pool = build_trial_pool(treatment, design, seed=seed, **pool_kwargs)
    return run_trials(pool, n_trials, seed=None if seed is None else seed + 1)
