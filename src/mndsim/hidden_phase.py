"""The hidden phase: biomarkers, compensation and the timing of treatment.

Motor neurons are lost for many months before symptoms because surviving
units reinnervate orphaned muscle fibres. Plasma neurofilament light (NfL)
reports the *rate* of axonal loss, while function reports the *state* left
after compensation. This module turns that distinction into three tools:

* :func:`response_lag` - after a disease-modifying treatment starts, how
  long until NfL falls, and how long until function visibly benefits? The
  model predicts that the functional lag is longest exactly when treatment is
  most valuable (early), which is why short functional endpoints can miss
  effective drugs that a biomarker detects.
* :func:`trigger_strategies` - in presymptomatic carriers, compare starting
  treatment when monitored NfL rises (as in prevention trials) against
  starting at symptom onset or at diagnosis.
* :func:`false_trigger_rate` - how often a monitoring rule fires in carriers
  who never convert, given assay noise and between-person variation.

All outputs are model predictions under illustrative parameters; they are
hypotheses to test, not clinical guidance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cohort import PopulationModel
from .config import DEFAULT_CONFIG, PhysiologyConfig
from .engine import healthy_nfl, simulate
from .metrics import time_to_threshold
from .patient import Patient
from .phenotypes import SOD1_FAMILIAL
from .result import SimulationResult
from .treatment import TreatmentLike, as_treatment_list, treatment_label


# ----------------------------------------------------------------------
# 1. Biomarker-function lag
# ----------------------------------------------------------------------
def standard_starts(result: SimulationResult, patient: int = 0) -> dict[str, float]:
    """Treatment start times (months since disease onset) at canonical stages."""
    onset = float(result.symptom_onset[patient])
    if not np.isfinite(onset):
        raise ValueError("patient never becomes symptomatic within the horizon")
    dx = onset + result.patients[patient].diagnostic_delay
    return {
        "presymptomatic (6 mo before symptoms)": max(onset - 6.0, 0.0),
        "at symptom onset": onset,
        "at diagnosis": dx,
        "1 year after diagnosis": dx + 12.0,
    }


def _first_time(t: np.ndarray, flag: np.ndarray) -> float:
    """First time at which ``flag`` is True (NaN if never)."""
    idx = np.flatnonzero(flag)
    return float(t[idx[0]]) if idx.size else float("nan")


def response_lag(
    patient: Patient,
    treatment: TreatmentLike,
    starts: Mapping[str, float] | None = None,
    *,
    months: float = 180.0,
    nfl_reduction: float = 0.15,
    function_gain: float = 1.0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Time from treatment start to a visible biomarker vs functional effect.

    For each start time, the treated and untreated trajectories of the same
    patient are compared:

    * ``nfl_lag``: months until treated NfL first falls at least
      ``nfl_reduction`` (fraction) below untreated NfL;
    * ``function_lag``: months until the functional score is first at least
      ``function_gain`` points better;
    * ``event_gain``: months gained before respiratory failure.

    Parameters
    ----------
    starts:
        ``{label: start time in months since disease onset}``; defaults to
        :func:`standard_starts` (presymptomatic, symptom onset, diagnosis,
        one year later).
    """
    base = simulate(patient, months=months, config=config)
    starts = dict(starts or standard_starts(base))
    treatments = as_treatment_list(treatment)
    rows = []
    for label, start in starts.items():
        tx = [t.at(start, max(t.stop, start)) for t in treatments]
        res = simulate(patient, tx, months=months, config=config)
        after = base.time >= start
        t = base.time[after] - start
        nfl_ratio = res.nfl[0, after] / base.nfl[0, after]
        gain = res.alsfrs[0, after] - base.alsfrs[0, after]
        rows.append(
            {
                "start": label,
                "start_month": start,
                "nfl_lag": _first_time(t, nfl_ratio <= 1.0 - nfl_reduction),
                "function_lag": _first_time(t, gain >= function_gain),
                "max_nfl_reduction": float(1.0 - np.nanmin(nfl_ratio)),
                "event_gain": float(res.event_time[0] - base.event_time[0]),
                "symptom_onset_delay": float(res.symptom_onset[0] - base.symptom_onset[0]),
            }
        )
    return pd.DataFrame(rows)


def lag_trajectories(
    patient: Patient,
    treatment: TreatmentLike,
    start: float,
    *,
    months: float = 180.0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Treated/untreated NfL ratio and functional difference over time since start."""
    base = simulate(patient, months=months, config=config)
    tx = [t.at(start, max(t.stop, start)) for t in as_treatment_list(treatment)]
    res = simulate(patient, tx, months=months, config=config)
    after = base.time >= start
    return pd.DataFrame(
        {
            "months_since_start": base.time[after] - start,
            "nfl_ratio": res.nfl[0, after] / base.nfl[0, after],
            "function_difference": res.alsfrs[0, after] - base.alsfrs[0, after],
            "mune_difference": (res.mu_count[0, after] - base.mu_count[0, after]).sum(-1),
        }
    )


# ----------------------------------------------------------------------
# 2. NfL-triggered treatment in presymptomatic carriers
# ----------------------------------------------------------------------
def carrier_population(genotype: str = "SOD1", base: Patient = SOD1_FAMILIAL) -> PopulationModel:
    """A population of gene carriers who will convert (for trigger studies)."""
    return PopulationModel(
        base=base,
        genotype_probs={genotype: 1.0},
        onset_region_probs={"lower_limb": 0.6, "upper_limb": 0.3, "bulbar": 0.1},
        onset_effects={},
        age_mean=50.0,
        age_sd=10.0,
    )


@dataclass(frozen=True)
class MonitoringRule:
    """How carriers are monitored and when treatment is triggered.

    Parameters
    ----------
    threshold:
        Trigger when measured NfL exceeds ``threshold`` x reference.
    interval:
        Months between blood draws.
    assay_cv:
        Within-person, visit-to-visit variability (log-normal CV), including
        assay error.
    between_person_cv:
        Variation of healthy NfL between people of the same age.
    monitor_before_onset:
        Months of monitoring before biological disease onset (carriers are
        followed for years before they convert).
    reference:
        ``"personal"`` (each person's own first two draws) or
        ``"population"`` (age-expected healthy value).
    confirm:
        Require two consecutive draws above threshold.
    """

    threshold: float = 2.0
    interval: float = 3.0
    assay_cv: float = 0.15
    between_person_cv: float = 0.3
    reference: str = "personal"
    confirm: bool = True
    monitor_before_onset: float = 24.0

    def __post_init__(self) -> None:
        if self.reference not in ("personal", "population"):
            raise ValueError("reference must be 'personal' or 'population'")
        if self.threshold <= 1.0 or self.interval <= 0:
            raise ValueError("threshold must be > 1 and interval > 0")


def _trigger_times(
    t: np.ndarray,
    nfl: np.ndarray,
    base: np.ndarray,
    rule: MonitoringRule,
    rng: np.random.Generator,
    monitor_from: np.ndarray | None = None,
) -> np.ndarray:
    """Per-patient trigger time under a monitoring rule (NaN if never)."""
    P = nfl.shape[0]
    person = np.exp(rule.between_person_cv * rng.standard_normal(P) - 0.5 * rule.between_person_cv**2)
    phase = rng.uniform(0.0, rule.interval, P)
    start = np.full(P, -rule.monitor_before_onset) if monitor_from is None else monitor_from
    out = np.full(P, np.nan)
    for p in range(P):
        visits = np.arange(start[p] + phase[p], t[-1], rule.interval)
        if visits.size < 3:
            continue
        true = np.interp(visits, t, nfl[p], left=base[p])  # healthy before disease onset
        healthy = base[p] * person[p]
        level = (true - base[p]) + healthy  # disease release on top of the personal baseline
        measured = level * np.exp(rule.assay_cv * rng.standard_normal(visits.size))
        if rule.reference == "personal":
            ref = measured[:2].mean()
            candidates = np.arange(2, visits.size)
        else:
            ref = base[p]
            candidates = np.arange(visits.size)
        above = measured[candidates] >= rule.threshold * ref
        if rule.confirm:
            ok = above[:-1] & above[1:]
            hit = np.flatnonzero(ok)
            if hit.size:
                out[p] = visits[candidates[hit[0] + 1]]
        else:
            hit = np.flatnonzero(above)
            if hit.size:
                out[p] = visits[candidates[hit[0]]]
    return out


@dataclass(frozen=True, eq=False)
class StrategyResult:
    """Outcomes of treatment-start strategies in a carrier cohort."""

    table: pd.DataFrame  # one row per patient x strategy
    rule: MonitoringRule
    treatment_name: str

    def summary(self) -> pd.DataFrame:
        """Median outcome of each strategy (months)."""
        g = self.table.groupby("strategy", sort=False)
        return pd.DataFrame(
            {
                "treated_fraction": g.started.mean(),
                "median_start_vs_symptoms": g.start_vs_symptoms.median(),
                "median_onset_delay": g.symptom_onset_delay.median(),
                "median_event_gain": g.event_gain.median(),
                "median_function_at_+24mo": g.alsfrs_24m_after_untreated_onset.median(),
            }
        )


def trigger_strategies(
    treatment: TreatmentLike,
    *,
    n: int = 300,
    population: PopulationModel | None = None,
    rule: MonitoringRule = MonitoringRule(),
    months: float = 200.0,
    seed: int | None = 0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
) -> StrategyResult:
    """Compare when to start a treatment in presymptomatic carriers.

    Strategies: ``untreated``, ``nfl_trigger`` (start at the visit that
    fires ``rule``), ``symptom_onset`` and ``diagnosis``. Outcomes per
    patient: delay of symptom onset, months gained before respiratory
    failure, and functional score 24 months after the untreated symptom
    onset. Pre-treatment trajectories are identical across strategies.
    """
    rng = np.random.default_rng(seed)
    pop = population or carrier_population()
    patients = pop.sample(n, seed=int(rng.integers(2**31)), name_prefix="carrier")
    base = simulate(patients, months=months, dt=0.5, config=config)
    onset = base.symptom_onset
    healthy = healthy_nfl(np.array([p.age_at_onset for p in patients]), config)
    trig = _trigger_times(base.time, base.nfl, healthy, rule, rng)
    starts = {
        "nfl_trigger": trig,
        "symptom_onset": onset,
        "diagnosis": onset + np.array([p.diagnostic_delay for p in patients]),
    }
    eval_t = onset + 24.0
    frames = []

    def add(name: str, res: SimulationResult, start: np.ndarray) -> None:
        frames.append(
            pd.DataFrame(
                {
                    "patient": [p.name for p in patients],
                    "strategy": name,
                    "started": np.isfinite(start),
                    "start_vs_symptoms": start - onset,
                    "symptom_onset_delay": res.symptom_onset - onset,
                    "event_gain": res.event_time - base.event_time,
                    "alsfrs_24m_after_untreated_onset": res.value_at("alsfrs", eval_t),
                }
            )
        )

    add("untreated", base, np.full(n, np.nan))
    for name, start in starts.items():
        origin = np.where(np.isfinite(start), start, np.inf)
        res = simulate(patients, treatment, months=months, dt=0.5, config=config, time_origin=origin)
        add(name, res, start)
    return StrategyResult(pd.concat(frames, ignore_index=True), rule, treatment_label(treatment))


def false_trigger_rate(
    rule: MonitoringRule = MonitoringRule(),
    *,
    n: int = 2000,
    years: float = 10.0,
    age: float = 45.0,
    seed: int | None = 0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
) -> float:
    """Fraction of never-converting carriers in whom the rule fires within ``years``.

    Healthy NfL rises slowly with age, so a population reference also
    produces age-driven triggers; a personal reference removes most of them.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 12.0 * years + 1.0, 1.0)
    ages = age + t / 12.0
    nfl = np.broadcast_to(healthy_nfl(ages, config), (n, t.size))
    base = np.full(n, float(healthy_nfl(age, config)))
    trig = _trigger_times(t, nfl, base, rule, rng)
    return float(np.mean(np.isfinite(trig)))


def threshold_tradeoff(
    thresholds: Sequence[float] = (1.3, 1.5, 1.75, 2.0, 2.5, 3.0),
    *,
    rule: MonitoringRule = MonitoringRule(),
    n: int = 300,
    population: PopulationModel | None = None,
    seed: int | None = 0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
    **kwargs: Any,
) -> pd.DataFrame:
    """Lead time before symptoms vs false-trigger rate across NfL thresholds."""
    rng = np.random.default_rng(seed)
    pop = population or carrier_population()
    patients = pop.sample(n, seed=int(rng.integers(2**31)), name_prefix="carrier")
    base = simulate(patients, months=200.0, dt=0.5, config=config)
    healthy = healthy_nfl(np.array([p.age_at_onset for p in patients]), config)
    rows = []
    for thr in thresholds:
        r = MonitoringRule(**{**rule.__dict__, "threshold": thr})
        trig = _trigger_times(base.time, base.nfl, healthy, r, np.random.default_rng(seed))
        lead = base.symptom_onset - trig
        rows.append(
            {
                "threshold": thr,
                "median_lead_time": float(np.nanmedian(lead)),
                "triggered_before_symptoms": float(np.mean(lead > 0)),
                "false_trigger_rate_10y": false_trigger_rate(r, seed=seed, config=config, **kwargs),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "MonitoringRule",
    "StrategyResult",
    "carrier_population",
    "false_trigger_rate",
    "lag_trajectories",
    "response_lag",
    "standard_starts",
    "threshold_tradeoff",
    "trigger_strategies",
]

# Re-exported for convenience in notebooks.
time_to_threshold = time_to_threshold
