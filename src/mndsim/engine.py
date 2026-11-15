"""Vectorised disease-progression engine.

State is held as arrays of shape ``(patients, regions, units)`` and
advanced with exact exponential transition probabilities over each time
step, so results are insensitive to ``dt`` within the stable range.

Each motor unit is in one of three states::

    healthy --(hazard lambda)--> denervating --(1/dying_back_duration)--> dead
                 ^                     |
                 +---(NMJ rescue)------+

* **healthy** units drive their fibres and adopt orphaned fibres by
  collateral sprouting, up to ``max_enlargement`` x their original size;
* **denervating** units ("dying-back") transmit unreliably, cannot sprout,
  and progressively shed fibres;
* **dead** units' remaining fibres are orphaned.

Orphaned fibres are either re-innervated or, if they wait too long, lost
irreversibly. Upper-motor-neuron loss reduces the maximal excitation of
each pool and, through excitotoxic drive, modulates LMN hazard.

In *mean-field* mode (default) the state variables are probabilities and
expected fibre counts, so a single run gives the expected trajectory. In
*stochastic* mode each unit's state is sampled, giving the step-wise
decrements seen in serial motor-unit number estimates.

Large batches are processed in cache-sized chunks of patients.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, PhysiologyConfig
from .patient import Patient, PatientBatch
from .pool import MotorUnitPool
from .regions import REGIONS, region_index
from .result import SimulationResult
from .scoring import fvc_percent, item_scores
from .treatment import Treatment, TreatmentLike, as_treatment_list, modifiers

_RESP = region_index("respiratory")
_SERIES_REGION = ("mvc", "mu_count", "cmap", "enlargement", "denervated", "lost_fibers", "umn")


def _unit_vulnerability(
    patients: Sequence[Patient],
    offset: int,
    pool: MotorUnitPool,
    config: PhysiologyConfig,
    seed: int | None,
) -> np.ndarray:
    """Draw per-unit vulnerability (P, R, N); reproducible per patient."""
    n_regions, n_units = len(REGIONS), pool.n_units
    type_mult = np.asarray(config.type_vulnerability)[pool.unit_type]
    sd = config.unit_vulnerability_sd
    out = np.empty((len(patients), n_regions, n_units))
    for k, patient in enumerate(patients):
        if patient.seed is not None:
            rng = np.random.default_rng(patient.seed)
        elif seed is not None:
            rng = np.random.default_rng([seed, offset + k])
        else:
            rng = np.random.default_rng()
        z = rng.standard_normal((n_regions, n_units))
        out[k] = type_mult * np.exp(sd * z - 0.5 * sd**2)
    return out


def _first_crossing(
    found: np.ndarray,
    out: np.ndarray,
    prev: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    t_prev: float,
    dt: float,
) -> np.ndarray:
    """Record (in place) the interpolated first time ``cur`` drops below threshold."""
    hit = (~found) & (cur < threshold)
    if np.any(hit):
        drop = prev[hit] - cur[hit]
        denom = np.where(drop > 0, drop, np.inf)
        frac = np.clip((prev[hit] - threshold) / denom, 0.0, 1.0)
        out[hit] = t_prev + frac * dt
        found |= hit
    return hit


@dataclass
class _Outputs:
    """Preallocated output arrays for the whole batch."""

    series: dict[str, np.ndarray]
    units_h: np.ndarray | None
    units_g: np.ndarray | None
    symptom_time: np.ndarray
    symptom_region: np.ndarray
    event_time: np.ndarray

    @classmethod
    def allocate(cls, P: int, n_rec: int, n_items: int, N: int, record_units: bool) -> _Outputs:
        R = len(REGIONS)
        series = {name: np.full((P, n_rec, R), np.nan) for name in _SERIES_REGION}
        series["type_survival"] = np.full((P, n_rec, R, 3), np.nan)
        series["items"] = np.full((P, n_rec, n_items), np.nan)
        units = (lambda: np.full((P, n_rec, R, N), np.nan, dtype=np.float32)) if record_units else None
        return cls(
            series=series,
            units_h=units() if units else None,
            units_g=units() if units else None,
            symptom_time=np.full(P, np.nan),
            symptom_region=np.full(P, -1, dtype=int),
            event_time=np.full(P, np.nan),
        )


def _run_chunk(
    out: _Outputs,
    sl: slice,
    batch: PatientBatch,
    treatments: list[Treatment],
    pool: MotorUnitPool,
    config: PhysiologyConfig,
    vulnerability: np.ndarray,
    origin: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    n_steps: int,
    record_steps: list[int],
    stochastic: bool,
    stop_time: np.ndarray,
    stop_after_onset: np.ndarray,
    stop_margin: float,
) -> None:
    P, R, N = len(batch), len(REGIONS), pool.n_units
    n_rec = len(record_steps)
    sprout_competence = np.asarray(config.type_sprouting)[pool.unit_type]
    capacity = batch.max_enlargement[:, None, None] * pool.fibers
    type_masks = [pool.unit_type == k for k in range(3)]
    type_counts = np.array([max(m.sum(), 1) for m in type_masks], dtype=float)
    series = {name: arr[sl] for name, arr in out.series.items()}
    units_h = out.units_h[sl] if out.units_h is not None else None
    units_g = out.units_g[sl] if out.units_g is not None else None

    pH = np.ones((P, R, N))
    pG = np.zeros((P, R, N))
    nH = np.broadcast_to(pool.fibers, (P, R, N)).copy()
    nG = np.zeros((P, R, N))
    orphaned = np.zeros((P, R))
    lost = np.zeros((P, R))
    umn = np.ones((P, R))

    symptom_found = np.zeros(P, dtype=bool)
    symptom_time = np.full(P, np.nan)
    symptom_region = np.full(P, -1, dtype=int)
    event_found = np.zeros(P, dtype=bool)
    event_time = np.full(P, np.nan)

    retract_q = -np.expm1(-dt / config.terminal_retraction_time)
    atrophy_q = -np.expm1(-dt / config.denervated_fiber_survival)
    death_rate = 1.0 / config.dying_back_duration
    g = config.denervating_transmission
    ramp = config.regional_ramp_time
    drive_floor = config.residual_drive

    def strength(mods) -> tuple[np.ndarray, np.ndarray]:
        excitation = pool.max_excitation * (drive_floor + (1.0 - drive_floor) * umn)
        fibers_eff = nH + g * nG
        return pool.fast_force(fibers_eff, excitation) * mods.force[:, None], fibers_eff

    def record(slot: int, mvc: np.ndarray, fibers_eff: np.ndarray) -> None:
        alive = pH + pG
        series["mvc"][:, slot] = mvc
        series["mu_count"][:, slot] = alive.sum(-1)
        series["cmap"][:, slot] = fibers_eff.sum(-1)
        own = (alive * pool.fibers).sum(-1)
        series["enlargement"][:, slot] = np.divide(
            (nH + nG).sum(-1), own, out=np.full(own.shape, np.nan), where=own > 1e-12
        )
        series["denervated"][:, slot] = orphaned
        series["lost_fibers"][:, slot] = lost
        series["umn"][:, slot] = umn
        for k, m in enumerate(type_masks):
            series["type_survival"][:, slot, :, k] = alive[..., m].sum(-1) / type_counts[k]
        series["items"][:, slot] = item_scores(mvc, config.functional_items)
        if units_h is not None:
            units_h[:, slot] = pH
            units_g[:, slot] = pG

    mods = modifiers(treatments, 0.0 - origin, batch.genotype)
    mvc, fibers_eff = strength(mods)
    prev_min = mvc.min(-1)
    prev_resp = mvc[:, _RESP]
    record(0, mvc, fibers_eff)
    slot = 1
    early_stop = bool(np.any(np.isfinite(stop_time)) or np.any(np.isfinite(stop_after_onset)))

    for step in range(1, n_steps + 1):
        t_prev = (step - 1) * dt

        # Regional disease activity: 0 before the disease arrives, ramping to 1.
        arrived = t_prev >= batch.regional_onset
        if ramp > 0:
            elapsed = np.maximum(t_prev - batch.regional_onset, 0.0)
            activity = np.where(arrived, -np.expm1(-elapsed / ramp), 0.0)
        else:
            activity = arrived.astype(float)

        # Excitotoxic drive from surviving UMNs, then UMN loss.
        drive = batch.excitotoxicity[:, None] * umn * mods.excitotoxic[:, None]
        umn = umn * np.exp(-batch.umn_rate * activity * mods.umn[:, None] * dt)

        # Per-unit probability of *staying* healthy over this step.
        regional_rate = batch.lmn_rate * activity * (1.0 + drive) * mods.lmn[:, None]
        keep = np.exp(vulnerability * (regional_rate * -dt)[..., None])
        exit_rate = death_rate + mods.rescue
        q_exit = -np.expm1(-exit_rate * dt)
        c_death = (q_exit * death_rate / exit_rate)[:, None]  # (P, 1): G -> dead
        c_back = (q_exit[:, None] - c_death)[..., None]  # (P, 1, 1): G -> healthy
        stay_g = (1.0 - q_exit)[:, None, None]  # (P, 1, 1)

        if stochastic:
            u = rng.random((P, R, N))
            go_hg = (pH > 0.5) & (u >= keep)
            leave = (pG > 0.5) & (u < 1.0 - stay_g)
            die = leave & (rng.random((P, R, N)) * (1.0 - stay_g) < c_death[..., None])
            back = leave & ~die
            f_hg = np.where(go_hg, nH, 0.0)
            f_gx = np.where(die, nG, 0.0)
            f_gh = np.where(back, nG, 0.0)
            pH = np.where(go_hg, 0.0, np.where(back, 1.0, pH))
            pG = np.where(go_hg, 1.0, np.where(leave, 0.0, pG))
            nH += f_gh - f_hg
            nG += f_hg - f_gx - f_gh
            dead_fibers = f_gx.sum(-1)
        else:
            # Exact expected flows; written to minimise full-size temporaries.
            dead_fibers = c_death * nG.sum(-1)
            h_kept = pH * keep
            pG_back = pG * c_back
            pG = pG * stay_g + (pH - h_kept)
            pH = h_kept + pG_back
            n_kept = nH * keep
            nG_back = nG * c_back
            nG = nG * stay_g + (nH - n_kept)
            nH = n_kept + nG_back

        # Terminal retraction of denervating units (dying-back).
        shed_total = retract_q * nG.sum(-1)
        nG *= 1.0 - retract_q
        orphaned = orphaned + dead_fibers + shed_total

        # Collateral reinnervation by healthy units with spare capacity.
        room = np.maximum(capacity * mods.capacity[:, None, None] * pH - nH, 0.0)
        weight = sprout_competence * room
        total = weight.sum(-1)
        rate = batch.sprouting_rate[:, None] * mods.sprouting[:, None] * total
        adopted = orphaned * -np.expm1(-rate * dt)
        per_weight = np.divide(adopted, total, out=np.zeros_like(total), where=total > 0)
        # Competence <= 1, so capping the per-weight share at 1 keeps every
        # unit's gain within its remaining room.
        per_weight = np.minimum(per_weight, 1.0)
        weight *= per_weight[..., None]
        nH += weight
        orphaned = orphaned - per_weight * total

        # Orphaned fibres that wait too long degenerate irreversibly.
        atrophied = orphaned * atrophy_q
        orphaned = orphaned - atrophied
        lost = lost + atrophied

        # Read-outs and events.
        t = step * dt
        mods = modifiers(treatments, t - origin, batch.genotype)
        mvc, fibers_eff = strength(mods)
        cur_min = mvc.min(-1)
        hit = _first_crossing(
            symptom_found, symptom_time, prev_min, cur_min, config.symptom_threshold, t_prev, dt
        )
        if np.any(hit):
            symptom_region[hit] = np.argmin(mvc[hit], axis=-1)
        cur_resp = mvc[:, _RESP]
        _first_crossing(
            event_found, event_time, prev_resp, cur_resp, config.respiratory_failure_threshold, t_prev, dt
        )
        prev_min, prev_resp = cur_min, cur_resp

        if slot < n_rec and step == record_steps[slot]:
            record(slot, mvc, fibers_eff)
            slot += 1
            if early_stop:
                limit = np.fmin(stop_time, symptom_time + stop_after_onset)
                if np.all(t >= limit + stop_margin):
                    break

    out.symptom_time[sl] = symptom_time
    out.symptom_region[sl] = symptom_region
    out.event_time[sl] = event_time


def simulate(
    patients: Patient | Sequence[Patient],
    treatment: TreatmentLike = None,
    *,
    months: float = 72.0,
    dt: float = 0.25,
    record_every: float = 1.0,
    stochastic: bool = False,
    seed: int | None = 0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
    time_origin: float | np.ndarray = 0.0,
    record_units: bool = False,
    chunk_size: int = 256,
    stop_time: float | np.ndarray | None = None,
    stop_after_onset: float | np.ndarray | None = None,
) -> SimulationResult:
    """Simulate disease progression for one or many patients.

    Parameters
    ----------
    patients:
        A :class:`~mndsim.Patient` or a sequence of them (simulated together,
        vectorised).
    treatment:
        ``None``, a :class:`~mndsim.Treatment`, a combination (``a + b``) or a
        list of treatments. Applied to every patient.
    months:
        Simulated duration, in months since disease onset.
    dt:
        Integration step (months).
    record_every:
        Spacing of recorded time points (months); rounded to a multiple of
        ``dt``.
    stochastic:
        Sample individual motor-unit fates instead of tracking expectations.
    seed:
        Seed for unit vulnerability (patients without their own ``seed``) and
        stochastic transitions. ``None`` gives non-reproducible runs.
    config:
        Physiological constants.
    time_origin:
        Per-patient (or scalar) time that treatment schedules are measured
        from, e.g. each patient's enrolment time in a trial.
    record_units:
        Also store the state of every motor unit (for raster plots).
    chunk_size:
        Patients integrated together; tune for cache size. Results do not
        depend on it in mean-field mode.
    stop_time, stop_after_onset:
        Optional per-patient early stopping, for speed. A chunk stops once
        every patient is past ``stop_time`` or past ``symptom onset +
        stop_after_onset``. Recorded values after that point are NaN.

    Returns
    -------
    SimulationResult
    """
    if months <= 0 or dt <= 0:
        raise ValueError("months and dt must be > 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    batch = PatientBatch.from_patients(patients)
    treatments = as_treatment_list(treatment)
    pool = MotorUnitPool.from_config(config)
    P = len(batch)

    n_steps = max(1, round(months / dt))
    stride = max(1, round(record_every / dt))
    record_steps = sorted(set(range(0, n_steps + 1, stride)) | {n_steps})
    times = np.array(record_steps, dtype=float) * dt

    def per_patient(value: float | np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.full(P, np.inf)
        return np.broadcast_to(np.asarray(value, dtype=float), (P,)).copy()

    origin = per_patient(time_origin)
    stop = per_patient(stop_time)
    after_onset = per_patient(stop_after_onset)

    out = _Outputs.allocate(P, len(record_steps), len(config.functional_items), pool.n_units, record_units)
    rng = np.random.default_rng(seed)
    for start in range(0, P, chunk_size):
        sl = slice(start, min(start + chunk_size, P))
        sub = batch.take(sl)
        vulnerability = _unit_vulnerability(sub.patients, start, pool, config, seed)
        _run_chunk(
            out,
            sl,
            sub,
            treatments,
            pool,
            config,
            vulnerability,
            origin[sl],
            rng,
            dt,
            n_steps,
            record_steps,
            stochastic,
            stop[sl],
            after_onset[sl],
            stop_margin=stride * dt,
        )

    series = out.series
    items = series.pop("items")
    return SimulationResult(
        time=times,
        patients=batch.patients,
        treatment=tuple(treatments),
        config=config,
        pool=pool,
        stochastic=stochastic,
        items=items,
        alsfrs=items.sum(-1),
        fvc=fvc_percent(series["mvc"][..., _RESP]),
        symptom_onset=out.symptom_time,
        symptom_region=out.symptom_region,
        event_time=out.event_time,
        time_origin=origin,
        units_healthy=out.units_h,
        units_denervating=out.units_g,
        **series,
    )
