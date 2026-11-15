"""Mechanism-based treatments.

A :class:`Treatment` is described by *what it does* to the disease
mechanics, not by a drug name. Effects are fractional and switch on with a
first-order ramp after ``start`` and wash out after ``stop``. Several
treatments can be combined; protections then compose multiplicatively.

All effect sizes in :data:`TREATMENTS` are illustrative archetypes for
hypothesis exploration and trial-design teaching. None represents the
measured efficacy of a real drug.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

_FRACTIONAL = ("lmn_protection", "umn_protection", "anti_excitotoxic")
_NONNEGATIVE = ("nmj_rescue", "sprouting_boost", "capacity_boost", "symptomatic")
#: Names of all effect parameters of a :class:`Treatment`.
EFFECTS: tuple[str, ...] = _FRACTIONAL + _NONNEGATIVE


@dataclass(frozen=True)
class Treatment:
    """A disease-modifying and/or symptomatic intervention.

    Parameters
    ----------
    name:
        Label used in tables and plots.
    lmn_protection:
        Fractional reduction of the lower-motor-neuron degeneration hazard
        (0-1).
    umn_protection:
        Fractional reduction of the upper-motor-neuron loss rate (0-1).
    anti_excitotoxic:
        Fractional reduction of glutamatergic excitotoxic drive (0-1).
    nmj_rescue:
        Rate (per month) at which denervating units are re-stabilised and
        returned to the healthy state (neuromuscular-junction stabilisation).
    sprouting_boost:
        Fractional increase of the collateral-reinnervation rate.
    capacity_boost:
        Fractional increase of the maximum size surviving units can reach
        by sprouting (``max_enlargement``).
    symptomatic:
        Fractional increase of force output with *no* effect on
        degeneration (e.g. improved neuromuscular transmission).
    start:
        Months (since disease onset, or since ``time_origin`` passed to
        :func:`mndsim.simulate`) at which treatment begins.
    stop:
        Months at which treatment is withdrawn (``inf`` = never).
    ramp:
        Time constant (months) for the effect to build up after ``start``.
    washout:
        Time constant (months) for the effect to decay after ``stop``.
    target_genotype:
        If given, only patients with this genotype respond.
    description:
        Free-text description.
    """

    name: str = "treatment"
    lmn_protection: float = 0.0
    umn_protection: float = 0.0
    anti_excitotoxic: float = 0.0
    nmj_rescue: float = 0.0
    sprouting_boost: float = 0.0
    capacity_boost: float = 0.0
    symptomatic: float = 0.0
    start: float = 0.0
    stop: float = math.inf
    ramp: float = 1.0
    washout: float = 1.0
    target_genotype: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        for name in _FRACTIONAL:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        for name in (*_NONNEGATIVE, "ramp", "washout"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.stop < self.start:
            raise ValueError("stop must be >= start")

    # ------------------------------------------------------------------
    def replace(self, **changes: Any) -> Treatment:
        return replace(self, **changes)

    def at(self, start: float, stop: float = math.inf) -> Treatment:
        """Return the same treatment scheduled to start (and stop) at given times."""
        return replace(self, start=start, stop=stop)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_null(self) -> bool:
        """True if the treatment has no effect at all (a placebo)."""
        return all(getattr(self, name) == 0.0 for name in (*_FRACTIONAL, *_NONNEGATIVE))

    def efficacy(self, t: np.ndarray | float) -> np.ndarray:
        """Fraction (0-1) of the full effect present at time(s) ``t``."""
        t = np.asarray(t, dtype=float)
        on = _first_order(t - self.start, self.ramp)
        if math.isinf(self.stop):
            return on
        at_stop = _first_order(np.asarray(self.stop - self.start), self.ramp)
        off = at_stop * np.exp(-np.maximum(t - self.stop, 0.0) / max(self.washout, 1e-12))
        return np.where(t < self.stop, on, np.where(self.washout == 0, 0.0, off))

    def __add__(self, other: Treatment | Combination) -> Combination:
        return Combination.of(self, other)


def _first_order(elapsed: np.ndarray, tau: float) -> np.ndarray:
    elapsed = np.asarray(elapsed, dtype=float)
    if tau == 0:
        return (elapsed >= 0).astype(float)
    return np.where(elapsed >= 0, -np.expm1(-np.maximum(elapsed, 0.0) / tau), 0.0)


@dataclass(frozen=True)
class Combination:
    """Several treatments given together."""

    treatments: tuple[Treatment, ...]

    @classmethod
    def of(cls, *items: Treatment | Combination) -> Combination:
        return cls(tuple(as_treatment_list(items)))

    @property
    def name(self) -> str:
        return " + ".join(t.name for t in self.treatments)

    def at(self, start: float, stop: float = math.inf) -> Combination:
        return Combination(tuple(t.at(start, stop) for t in self.treatments))

    def __add__(self, other: Treatment | Combination) -> Combination:
        return Combination.of(self, other)


TreatmentLike = Treatment | Combination | Sequence[Treatment] | None


def as_treatment_list(treatment: TreatmentLike | Iterable) -> list[Treatment]:
    """Normalise ``None``/Treatment/Combination/sequence to a flat list."""
    if treatment is None:
        return []
    if isinstance(treatment, Treatment):
        return [treatment]
    if isinstance(treatment, Combination):
        return list(treatment.treatments)
    out: list[Treatment] = []
    for item in treatment:
        out.extend(as_treatment_list(item))
    return out


def treatment_label(treatment: TreatmentLike) -> str:
    items = as_treatment_list(treatment)
    return " + ".join(t.name for t in items) if items else "untreated"


@dataclass(frozen=True)
class Modifiers:
    """Per-patient multipliers applied by treatments at one time step."""

    lmn: np.ndarray
    umn: np.ndarray
    excitotoxic: np.ndarray
    rescue: np.ndarray
    sprouting: np.ndarray
    capacity: np.ndarray
    force: np.ndarray


def modifiers(
    treatments: Sequence[Treatment],
    t: np.ndarray,
    genotype: np.ndarray,
) -> Modifiers:
    """Combine treatments into per-patient multipliers at patient-time ``t``."""
    shape = np.shape(t)
    lmn = np.ones(shape)
    umn = np.ones(shape)
    exc = np.ones(shape)
    rescue = np.zeros(shape)
    sprout = np.ones(shape)
    capacity = np.ones(shape)
    force = np.ones(shape)
    for tx in treatments:
        e = tx.efficacy(t)
        if tx.target_genotype is not None:
            e = e * (genotype == tx.target_genotype)
        lmn *= 1.0 - e * tx.lmn_protection
        umn *= 1.0 - e * tx.umn_protection
        exc *= 1.0 - e * tx.anti_excitotoxic
        rescue += e * tx.nmj_rescue
        sprout *= 1.0 + e * tx.sprouting_boost
        capacity *= 1.0 + e * tx.capacity_boost
        force *= 1.0 + e * tx.symptomatic
    return Modifiers(lmn, umn, exc, rescue, sprout, capacity, force)


# ----------------------------------------------------------------------
# Library of mechanism archetypes (illustrative effect sizes)
# ----------------------------------------------------------------------

PLACEBO = Treatment(name="placebo", description="No pharmacological effect.")

NEUROPROTECTIVE = Treatment(
    name="neuroprotective",
    lmn_protection=0.35,
    umn_protection=0.35,
    description="Broad neuroprotection: slows both LMN and UMN degeneration.",
)

ANTI_EXCITOTOXIC = Treatment(
    name="anti_excitotoxic",
    anti_excitotoxic=0.6,
    description="Reduces glutamatergic excitotoxic drive onto lower motor neurons.",
)

NMJ_STABILIZER = Treatment(
    name="nmj_stabilizer",
    nmj_rescue=0.15,
    description="Re-stabilises retracting neuromuscular junctions (anti dying-back).",
)

SPROUTING_ENHANCER = Treatment(
    name="sprouting_enhancer",
    sprouting_boost=1.0,
    capacity_boost=0.4,
    description="Faster collateral reinnervation and 40% larger maximal unit size.",
)

SYMPTOMATIC = Treatment(
    name="symptomatic",
    symptomatic=0.10,
    ramp=0.25,
    washout=0.25,
    description="Boosts neuromuscular output by 10% without modifying degeneration.",
)

GENE_TARGETED = Treatment(
    name="gene_targeted",
    lmn_protection=0.7,
    umn_protection=0.5,
    ramp=3.0,
    target_genotype="SOD1",
    description="Large effect restricted to one genetic subgroup (here: SOD1).",
)

TREATMENTS: dict[str, Treatment] = {
    t.name: t
    for t in (
        PLACEBO,
        NEUROPROTECTIVE,
        ANTI_EXCITOTOXIC,
        NMJ_STABILIZER,
        SPROUTING_ENHANCER,
        SYMPTOMATIC,
        GENE_TARGETED,
    )
}


def get_treatment(name: str) -> Treatment:
    try:
        return TREATMENTS[name]
    except KeyError:
        raise KeyError(f"unknown treatment {name!r}; available: {sorted(TREATMENTS)}") from None
