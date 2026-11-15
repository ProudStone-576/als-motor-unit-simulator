"""Patient (virtual subject) parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import numpy as np

from .regions import NEURAXIS_POSITION, REGIONS, region_index

#: Parameters that are rates/scales and therefore sampled and fitted on a log scale.
POSITIVE_PARAMETERS: tuple[str, ...] = (
    "lmn_rate",
    "umn_rate",
    "excitotoxicity",
    "spread_rate",
    "sprouting_rate",
    "max_enlargement",
)


@dataclass(frozen=True)
class Patient:
    """Disease parameters of one virtual patient.

    Time is measured in **months since biological disease onset** in the
    first affected region. Symptom onset is *emergent*: it happens when
    compensation by collateral reinnervation can no longer hide motor-unit
    loss (see :attr:`mndsim.SimulationResult.symptom_onset`).

    Parameters
    ----------
    name:
        Label used in tables and plots.
    onset_region:
        Region where degeneration starts (one of :data:`mndsim.REGIONS`).
    lmn_rate:
        Baseline hazard (per month) that a fast fatigue-resistant lower motor
        neuron enters the denervating state, before excitotoxic
        amplification. S and FF units scale it by their type vulnerability.
    umn_rate:
        Exponential loss rate (per month) of upper motor neurons.
    excitotoxicity:
        Amplification of LMN hazard by glutamatergic drive from surviving
        upper motor neurons (0 = none; 1 doubles the hazard while all UMNs
        survive). Encodes the corticomotoneuronal 'dying-forward' hypothesis.
    spread_rate:
        Speed of contiguous spread along the neuraxis (position units per
        month, see :data:`mndsim.regions.NEURAXIS_POSITION`). ``0`` keeps the
        disease confined to the onset region.
    sprouting_rate:
        Rate constant of collateral reinnervation of orphaned muscle fibres.
    max_enlargement:
        Maximum size a surviving motor unit can reach through sprouting, as
        a multiple of its original fibre count.
    region_severity:
        Per-region multiplier of LMN and UMN rates (mapping or sequence in
        :data:`mndsim.REGIONS` order). ``0`` spares a region entirely.
    genotype:
        Free-text genetic label (e.g. ``"SOD1"``). Used by genotype-targeted
        treatments and cohort summaries; it has no effect on its own.
    age_at_onset:
        Age in years; informational (cohort covariate).
    diagnostic_delay:
        Months from symptom onset to diagnosis. Not part of the disease
        mechanics; used for the progression rate at diagnosis (Delta-FS) and
        for trial enrolment.
    seed:
        Seed for the unit-to-unit vulnerability draw, making the patient's
        motor-unit pools reproducible. ``None`` uses the simulation seed.
    """

    name: str = "patient"
    onset_region: str = "upper_limb"
    lmn_rate: float = 0.08
    umn_rate: float = 0.01
    excitotoxicity: float = 0.5
    spread_rate: float = 0.04
    sprouting_rate: float = 2.0
    max_enlargement: float = 5.0
    region_severity: tuple[float, ...] = field(default=(1.0, 1.0, 1.0, 1.0))
    genotype: str = "sporadic"
    age_at_onset: float = 60.0
    diagnostic_delay: float = 12.0
    seed: int | None = None

    def __post_init__(self) -> None:
        region_index(self.onset_region)
        severity = self.region_severity
        if isinstance(severity, Mapping):
            unknown = set(severity) - set(REGIONS)
            if unknown:
                raise ValueError(f"unknown regions in region_severity: {sorted(unknown)}")
            severity = tuple(float(severity.get(r, 1.0)) for r in REGIONS)
        else:
            severity = tuple(float(s) for s in severity)
        if len(severity) != len(REGIONS):
            raise ValueError(f"region_severity needs {len(REGIONS)} values")
        if any(s < 0 or not math.isfinite(s) for s in severity):
            raise ValueError("region_severity values must be finite and >= 0")
        object.__setattr__(self, "region_severity", severity)

        for name in ("lmn_rate", "umn_rate", "excitotoxicity", "spread_rate", "sprouting_rate"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if not (math.isfinite(self.diagnostic_delay) and self.diagnostic_delay >= 0):
            raise ValueError("diagnostic_delay must be finite and >= 0")
        if not (math.isfinite(self.max_enlargement) and self.max_enlargement >= 1.0):
            raise ValueError("max_enlargement must be >= 1")

    # ------------------------------------------------------------------
    def replace(self, **changes: Any) -> Patient:
        """Return a copy with some parameters changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["region_severity"] = dict(zip(REGIONS, self.region_severity, strict=True))
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Patient:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown Patient fields: {sorted(unknown)}")
        return cls(**dict(data))

    def regional_onset_times(self) -> np.ndarray:
        """Months after disease onset at which degeneration reaches each region."""
        origin = NEURAXIS_POSITION[self.onset_region]
        distance = np.array([abs(NEURAXIS_POSITION[r] - origin) for r in REGIONS])
        if self.spread_rate > 0:
            times = distance / self.spread_rate
        else:
            times = np.where(distance == 0, 0.0, np.inf)
        return times


@dataclass(frozen=True)
class PatientBatch:
    """Structure-of-arrays view of many patients, used by the vectorised engine."""

    patients: tuple[Patient, ...]
    lmn_rate: np.ndarray  # (P, R)
    umn_rate: np.ndarray  # (P, R)
    excitotoxicity: np.ndarray  # (P,)
    sprouting_rate: np.ndarray  # (P,)
    max_enlargement: np.ndarray  # (P,)
    regional_onset: np.ndarray  # (P, R)
    genotype: np.ndarray  # (P,) object

    @classmethod
    def from_patients(cls, patients: Patient | Sequence[Patient]) -> PatientBatch:
        if isinstance(patients, Patient):
            patients = (patients,)
        patients = tuple(patients)
        if not patients:
            raise ValueError("at least one patient is required")
        severity = np.array([p.region_severity for p in patients], dtype=float)
        return cls(
            patients=patients,
            lmn_rate=np.array([p.lmn_rate for p in patients])[:, None] * severity,
            umn_rate=np.array([p.umn_rate for p in patients])[:, None] * severity,
            excitotoxicity=np.array([p.excitotoxicity for p in patients], dtype=float),
            sprouting_rate=np.array([p.sprouting_rate for p in patients], dtype=float),
            max_enlargement=np.array([p.max_enlargement for p in patients], dtype=float),
            regional_onset=np.array([p.regional_onset_times() for p in patients]),
            genotype=np.array([p.genotype for p in patients], dtype=object),
        )

    def __len__(self) -> int:
        return len(self.patients)

    def take(self, index: slice | np.ndarray) -> PatientBatch:
        """Sub-batch for a slice or integer index array."""
        idx = np.arange(len(self))[index]
        return PatientBatch(
            patients=tuple(self.patients[i] for i in idx),
            lmn_rate=self.lmn_rate[idx],
            umn_rate=self.umn_rate[idx],
            excitotoxicity=self.excitotoxicity[idx],
            sprouting_rate=self.sprouting_rate[idx],
            max_enlargement=self.max_enlargement[idx],
            regional_onset=self.regional_onset[idx],
            genotype=self.genotype[idx],
        )
