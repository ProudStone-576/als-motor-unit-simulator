"""Heterogeneous virtual cohorts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .patient import Patient
from .phenotypes import TYPICAL
from .regions import REGIONS


def _default_onset() -> dict[str, float]:
    return {"upper_limb": 0.36, "lower_limb": 0.34, "bulbar": 0.28, "respiratory": 0.02}


def _default_genotypes() -> dict[str, float]:
    return {"sporadic": 0.90, "C9orf72": 0.08, "SOD1": 0.02}


def _default_log_sd() -> dict[str, float]:
    return {
        "lmn_rate": 0.60,
        "umn_rate": 0.70,
        "excitotoxicity": 0.50,
        "spread_rate": 0.60,
        "sprouting_rate": 0.30,
        "max_enlargement": 0.12,
    }


def _default_onset_effects() -> dict[str, dict[str, float]]:
    return {"bulbar": {"lmn_rate": 1.25, "spread_rate": 1.2}}


@dataclass(frozen=True)
class PopulationModel:
    """Statistical model of between-patient heterogeneity.

    Each virtual patient starts from ``base`` and has its rate parameters
    multiplied by log-normal factors. A shared latent "aggressiveness"
    factor correlates LMN hazard and spread rate (``aggressiveness_corr``).
    Onset-region and genotype frequencies are approximate figures for
    population-based cohorts of European ancestry; treat them as editable
    defaults.

    Parameters
    ----------
    base:
        Median patient.
    onset_region_probs, genotype_probs:
        Categorical frequencies (normalised automatically).
    log_sd:
        Log-scale SD of the multiplicative variation of each parameter.
    aggressiveness_corr:
        Correlation, on the log scale, between ``lmn_rate`` and
        ``spread_rate`` induced by the shared factor.
    onset_effects:
        Extra multipliers by onset region (bulbar onset progresses faster).
    age_mean, age_sd:
        Age at onset distribution (years); bulbar onset is ``bulbar_age_shift``
        years older on average.
    diagnostic_delay_median, diagnostic_delay_log_sd:
        Log-normal months from symptom onset to diagnosis.
    """

    base: Patient = TYPICAL
    onset_region_probs: Mapping[str, float] = field(default_factory=_default_onset)
    genotype_probs: Mapping[str, float] = field(default_factory=_default_genotypes)
    log_sd: Mapping[str, float] = field(default_factory=_default_log_sd)
    aggressiveness_corr: float = 0.6
    onset_effects: Mapping[str, Mapping[str, float]] = field(default_factory=_default_onset_effects)
    age_mean: float = 61.0
    age_sd: float = 11.0
    bulbar_age_shift: float = 5.0
    diagnostic_delay_median: float = 11.0
    diagnostic_delay_log_sd: float = 0.5

    def __post_init__(self) -> None:
        for region in self.onset_region_probs:
            if region not in REGIONS:
                raise ValueError(f"unknown onset region {region!r}")
        if not 0.0 <= self.aggressiveness_corr <= 1.0:
            raise ValueError("aggressiveness_corr must be in [0, 1]")

    def replace(self, **changes: Any) -> PopulationModel:
        return replace(self, **changes)

    def sample(self, n: int, seed: int | None = None, name_prefix: str = "vp") -> list[Patient]:
        """Draw ``n`` virtual patients."""
        if n < 1:
            raise ValueError("n must be >= 1")
        rng = np.random.default_rng(seed)
        regions = list(self.onset_region_probs)
        probs = np.array([self.onset_region_probs[r] for r in regions], dtype=float)
        onset = rng.choice(regions, size=n, p=probs / probs.sum())
        genotypes = list(self.genotype_probs)
        gprobs = np.array([self.genotype_probs[g] for g in genotypes], dtype=float)
        genotype = rng.choice(genotypes, size=n, p=gprobs / gprobs.sum())

        shared = rng.standard_normal(n)
        rho = np.sqrt(self.aggressiveness_corr)
        factors: dict[str, np.ndarray] = {}
        for name, sd in self.log_sd.items():
            z = rng.standard_normal(n)
            if name in ("lmn_rate", "spread_rate"):
                z = rho * shared + np.sqrt(1.0 - rho**2) * z
            factors[name] = np.exp(sd * z)

        age = rng.normal(self.age_mean, self.age_sd, n)
        age = np.where(onset == "bulbar", age + self.bulbar_age_shift, age)
        age = np.clip(age, 20.0, 90.0)
        delay = self.diagnostic_delay_median * np.exp(self.diagnostic_delay_log_sd * rng.standard_normal(n))
        seeds = rng.integers(0, 2**31 - 1, size=n)

        patients = []
        for k in range(n):
            params: dict[str, Any] = {}
            for name, f in factors.items():
                value = getattr(self.base, name) * f[k]
                value *= self.onset_effects.get(str(onset[k]), {}).get(name, 1.0)
                params[name] = max(value, 1.0) if name == "max_enlargement" else value
            patients.append(
                self.base.replace(
                    name=f"{name_prefix}{k:05d}",
                    onset_region=str(onset[k]),
                    genotype=str(genotype[k]),
                    age_at_onset=float(age[k]),
                    diagnostic_delay=float(delay[k]),
                    seed=int(seeds[k]),
                    **params,
                )
            )
        return patients


DEFAULT_POPULATION = PopulationModel()


def sample_cohort(
    n: int, seed: int | None = None, population: PopulationModel = DEFAULT_POPULATION
) -> list[Patient]:
    """Convenience wrapper around :meth:`PopulationModel.sample`."""
    return population.sample(n, seed=seed)
