"""Simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .metrics import interpolate_at, time_to_threshold
from .regions import ITEM_NAMES, REGIONS, region_index

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .config import PhysiologyConfig
    from .patient import Patient
    from .pool import MotorUnitPool
    from .treatment import Treatment


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Trajectories for a batch of patients.

    Array attributes have shape ``(patients, time, ...)``; the trailing axes
    are regions (in :data:`mndsim.REGIONS` order), functional items or unit
    types. Times are months since disease onset.

    Attributes
    ----------
    mvc:
        Maximal voluntary force per region (1 = healthy).
    mu_count:
        Surviving motor units per region (true value; healthy = ``n_units``).
    cmap:
        Compound muscle action potential amplitude per region, as a fraction
        of baseline (innervated fibres, weighted by transmission reliability).
    enlargement:
        Mean size of surviving units relative to their own original size
        (fibres innervated / fibres originally owned); >1 reflects collateral
        reinnervation.
    denervated:
        Fraction of muscle fibres currently denervated but viable (the
        substrate of fibrillation potentials on needle EMG).
    lost_fibers:
        Fraction of fibres irreversibly lost after prolonged denervation.
    umn:
        Surviving upper-motor-neuron fraction per region.
    type_survival:
        Surviving fraction of S, FR and FF units per region.
    items, alsfrs:
        Expected synthetic functional item scores (0-4) and total (0-48).
    fvc:
        Respiratory capacity, % predicted.
    nfl:
        Plasma neurofilament light chain (pg/mL, illustrative scale): released
        by degenerating lower and upper motor neuron axons in proportion to
        their size and length, and cleared from blood. It reports the *rate*
        of neurodegeneration, not the accumulated damage.
    symptom_onset:
        Months from disease onset to first noticeable weakness (NaN if never).
    symptom_region:
        Index of the region where symptoms began (-1 if never).
    event_time:
        Months from disease onset to respiratory failure, the composite
        'death or permanent ventilation' endpoint (NaN if not reached).
    """

    time: np.ndarray
    patients: tuple[Patient, ...]
    treatment: tuple[Treatment, ...]
    config: PhysiologyConfig
    pool: MotorUnitPool
    stochastic: bool
    mvc: np.ndarray
    mu_count: np.ndarray
    cmap: np.ndarray
    enlargement: np.ndarray
    denervated: np.ndarray
    lost_fibers: np.ndarray
    umn: np.ndarray
    type_survival: np.ndarray
    items: np.ndarray
    alsfrs: np.ndarray
    fvc: np.ndarray
    nfl: np.ndarray
    symptom_onset: np.ndarray
    symptom_region: np.ndarray
    event_time: np.ndarray
    time_origin: np.ndarray
    units_healthy: np.ndarray | None = None
    units_denervating: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def n_patients(self) -> int:
        return len(self.patients)

    def __len__(self) -> int:
        return self.n_patients

    def __repr__(self) -> str:
        tx = ", ".join(t.name for t in self.treatment) or "untreated"
        mode = "stochastic" if self.stochastic else "mean-field"
        return (
            f"SimulationResult({self.n_patients} patient(s), {self.time[-1]:g} months, "
            f"{len(self.time)} time points, {mode}, treatment: {tx})"
        )

    @property
    def mup_size(self) -> np.ndarray:
        """Mean motor-unit potential size relative to healthy (reinnervation enlarges units)."""
        n = self.pool.n_units
        with np.errstate(divide="ignore", invalid="ignore"):
            size = self.cmap * n / self.mu_count
        return np.where(self.mu_count > 1e-9, size, np.nan)

    @property
    def max_score(self) -> float:
        return 4.0 * self.items.shape[-1]

    @property
    def mu_loss(self) -> np.ndarray:
        """Fraction of motor units lost per region."""
        return 1.0 - self.mu_count / self.pool.n_units

    @property
    def survival_from_symptom_onset(self) -> np.ndarray:
        """Months from symptom onset to the respiratory-failure event."""
        return self.event_time - self.symptom_onset

    def region(self, name: str) -> int:
        return region_index(name)

    # ------------------------------------------------------------------
    def subset(self, index: Any) -> SimulationResult:
        """Return a result restricted to some patients (index, slice or mask)."""
        idx = np.arange(self.n_patients)[index]
        idx = np.atleast_1d(idx)
        fields: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "patients":
                fields[name] = tuple(self.patients[i] for i in idx)
            elif isinstance(value, np.ndarray) and name != "time" and value.shape[:1] == (self.n_patients,):
                fields[name] = value[idx]
            else:
                fields[name] = value
        return SimulationResult(**fields)

    def aligned_time(self, patient: int = 0) -> np.ndarray:
        """Time axis in months since this patient's symptom onset."""
        return self.time - self.symptom_onset[patient]

    def value_at(self, name: str, t: np.ndarray | float, region: str | None = None) -> np.ndarray:
        """Interpolate a recorded series at per-patient times ``t`` (shape (P,))."""
        series = getattr(self, name)
        if region is not None:
            series = series[..., region_index(region)]
        if series.ndim != 2:
            raise ValueError(f"{name!r} needs a region to be selected")
        t = np.broadcast_to(np.asarray(t, dtype=float), (self.n_patients,))
        return np.array([interpolate_at(self.time, series[p], t[p]) for p in range(self.n_patients)])

    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """One row per patient with the key clinical and electrophysiological metrics."""
        import pandas as pd

        onset = self.symptom_onset
        diagnosis = onset + np.array([p.diagnostic_delay for p in self.patients])
        score_dx = self.value_at("alsfrs", diagnosis)
        region_names = np.array([REGIONS[i] if i >= 0 else None for i in self.symptom_region])
        score_onset = self.value_at("alsfrs", onset)
        score_12 = self.value_at("alsfrs", onset + 12.0)
        loss_at_onset = np.full(self.n_patients, np.nan)
        cmap_at_onset = np.full(self.n_patients, np.nan)
        for p in range(self.n_patients):
            r = self.symptom_region[p]
            if r >= 0:
                loss_at_onset[p] = 1.0 - interpolate_at(self.time, self.mu_count[p, :, r], onset[p]) / (
                    self.pool.n_units
                )
                cmap_at_onset[p] = interpolate_at(self.time, self.cmap[p, :, r], onset[p])
        nfl_dx = self.value_at("nfl", diagnosis)
        baseline = self.nfl[:, 0]
        lead = np.array(
            [
                onset[p] - time_to_threshold(self.time, self.nfl[p], 2.0 * baseline[p], direction="above")
                for p in range(self.n_patients)
            ]
        )
        t_half = np.array(
            [time_to_threshold(self.time, self.alsfrs[p], 24.0) for p in range(self.n_patients)]
        )
        return pd.DataFrame(
            {
                "patient": [p.name for p in self.patients],
                "onset_region": [p.onset_region for p in self.patients],
                "genotype": [p.genotype for p in self.patients],
                "treatment": ", ".join(t.name for t in self.treatment) or "untreated",
                "symptom_onset_months": onset,
                "symptom_region": region_names,
                "mu_loss_at_symptom_onset": loss_at_onset,
                "cmap_at_symptom_onset": cmap_at_onset,
                "alsfrs_slope_first_year": (score_12 - score_onset) / 12.0,
                "diagnosis_months": diagnosis,
                "alsfrs_at_diagnosis": score_dx,
                "delta_fs": (self.max_score - score_dx) / (diagnosis - onset),
                "nfl_at_diagnosis": nfl_dx,
                "nfl_fold_at_diagnosis": nfl_dx / baseline,
                "nfl_lead_time": lead,
                "event_months": self.event_time,
                "survival_from_symptom_onset": self.survival_from_symptom_onset,
                "months_to_half_function": t_half,
                "final_alsfrs": self.alsfrs[:, -1],
            }
        )

    def to_frame(self, patient: int | None = None) -> pd.DataFrame:
        """Tidy wide table: one row per (patient, time) with every scalar series."""
        import pandas as pd

        patients = range(self.n_patients) if patient is None else [patient]
        frames = []
        mup = self.mup_size
        for p in patients:
            data: dict[str, Any] = {
                "patient": self.patients[p].name,
                "patient_index": p,
                "month": self.time,
                "months_since_symptom_onset": self.time - self.symptom_onset[p],
                "alsfrs": self.alsfrs[p],
                "fvc": self.fvc[p],
            }
            for r, region in enumerate(REGIONS):
                data[f"mvc_{region}"] = self.mvc[p, :, r]
                data[f"mu_count_{region}"] = self.mu_count[p, :, r]
                data[f"cmap_{region}"] = self.cmap[p, :, r]
                data[f"mup_size_{region}"] = mup[p, :, r]
                data[f"enlargement_{region}"] = self.enlargement[p, :, r]
                data[f"denervated_{region}"] = self.denervated[p, :, r]
                data[f"umn_{region}"] = self.umn[p, :, r]
            for k, item in enumerate(ITEM_NAMES):
                data[f"item_{item}"] = self.items[p, :, k]
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)

    def to_csv(self, path: str, **kwargs: Any) -> None:
        self.to_frame().to_csv(path, index=False, **kwargs)

    # ------------------------------------------------------------------
    def plot(self, patient: int = 0, **kwargs: Any):
        """Multi-panel overview figure (see :func:`mndsim.plotting.plot_result`)."""
        from .plotting import plot_result

        return plot_result(self, patient=patient, **kwargs)
