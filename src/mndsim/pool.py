"""Motor-unit pool organised by Henneman's size principle.

Implements the steady-state part of the Fuglevand, Winter & Patla (1993)
motor-unit pool model: for a given level of excitatory drive, each unit is
recruited once the drive exceeds its threshold, fires at a rate that
increases linearly with drive up to a peak, and contributes a mean force
given by its twitch summed through a sigmoidal force-frequency relation.

Only *steady* force matters on the months-long timescale of
neurodegeneration, so individual twitch trains are never simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .config import DEFAULT_CONFIG, UNIT_TYPES, PhysiologyConfig

_E = np.e
_FUSION_KNEE = 0.4  # normalised rate (contraction time x rate) where summation saturates
_FUSION_NORM = 1.0 - np.exp(-2.0 * _FUSION_KNEE**3)
_TABLE_SIZE = 2049


def mean_force_factor(normalised_rate: np.ndarray) -> np.ndarray:
    """Mean force of a unit, in multiples of its twitch amplitude.

    ``normalised_rate`` is contraction time (s) x firing rate (Hz). Below
    0.4, twitches sum linearly (mean force = twitch area x rate = e*x);
    above it, force saturates sigmoidally towards fused tetanus
    (Fuglevand et al., 1993). The function is continuous at the knee.
    """
    x = np.asarray(normalised_rate, dtype=float)
    linear = _E * x
    saturating = _E * _FUSION_KNEE * (1.0 - np.exp(-2.0 * x**3)) / _FUSION_NORM
    return np.where(x <= _FUSION_KNEE, linear, saturating)


@dataclass(frozen=True)
class MotorUnitPool:
    """Static properties of one regional motor-unit pool.

    Attributes are arrays of length ``n_units`` ordered from the smallest
    (first recruited) to the largest unit.

    Attributes
    ----------
    fibers:
        Fraction of the muscle's fibres innervated by each unit in health
        (sums to 1). Proportional to twitch force.
    threshold:
        Recruitment threshold, in arbitrary excitation units.
    peak_rate:
        Peak firing rate (Hz).
    contraction_time:
        Twitch contraction time (s).
    unit_type:
        0 = S, 1 = FR, 2 = FF.
    max_excitation:
        Excitation at which every unit fires at its peak rate.
    """

    fibers: np.ndarray
    threshold: np.ndarray
    peak_rate: np.ndarray
    contraction_time: np.ndarray
    unit_type: np.ndarray
    max_excitation: float
    min_firing_rate: float
    excitation_gain: float

    @classmethod
    def from_config(cls, config: PhysiologyConfig = DEFAULT_CONFIG) -> MotorUnitPool:
        n = config.n_units
        i = np.arange(1, n + 1, dtype=float)
        twitch = np.exp(np.log(config.twitch_force_range) * i / n)
        threshold = np.exp(np.log(config.recruitment_range) * i / n)
        peak_rate = config.peak_firing_rate_first - config.peak_firing_rate_diff * threshold / threshold[-1]
        # T_i = T_L * (1/P_i)^(1/c), with c = log_RT(RP), so T spans [T_L/RT, T_L].
        scale = np.log(config.twitch_force_range) / np.log(config.contraction_time_range)
        contraction_time = config.longest_contraction_time * (twitch / twitch[0]) ** (-1.0 / scale)

        cumulative = np.cumsum(config.type_fractions)
        rank = (i - 0.5) / n
        unit_type = np.searchsorted(cumulative, rank, side="right").clip(0, 2)

        max_excitation = threshold[-1] + (peak_rate[-1] - config.min_firing_rate) / config.excitation_gain

        return cls(
            fibers=twitch / twitch.sum(),
            threshold=threshold,
            peak_rate=peak_rate,
            contraction_time=contraction_time,
            unit_type=unit_type.astype(np.int8),
            max_excitation=float(max_excitation),
            min_firing_rate=config.min_firing_rate,
            excitation_gain=config.excitation_gain,
        )

    @property
    def n_units(self) -> int:
        return int(self.fibers.size)

    @property
    def type_names(self) -> tuple[str, ...]:
        return UNIT_TYPES

    def firing_rates(self, excitation: np.ndarray | float) -> np.ndarray:
        """Firing rate (Hz) of every unit; shape ``excitation.shape + (n_units,)``."""
        e = np.asarray(excitation, dtype=float)[..., None]
        rate = self.excitation_gain * (e - self.threshold) + self.min_firing_rate
        rate = np.minimum(rate, self.peak_rate)
        return np.where(e >= self.threshold, rate, 0.0)

    def force_per_fiber(self, excitation: np.ndarray | float) -> np.ndarray:
        """Mean force produced per unit of innervated fibre, for every unit."""
        x = self.firing_rates(excitation) * self.contraction_time
        return mean_force_factor(x)

    @cached_property
    def healthy_max_force(self) -> float:
        """Force of the intact pool at maximal excitation (normalisation constant)."""
        return float(np.sum(self.fibers * self.force_per_fiber(self.max_excitation)))

    def force(self, fibers: np.ndarray, excitation: np.ndarray | float) -> np.ndarray:
        """Normalised force (1 = healthy maximum) of a pool with given fibre counts.

        ``fibers`` has shape ``(..., n_units)`` and ``excitation`` broadcasts
        against ``fibers.shape[:-1]``.
        """
        per_fiber = self.force_per_fiber(excitation)
        return np.sum(fibers * per_fiber, axis=-1) / self.healthy_max_force

    @cached_property
    def _force_table(self) -> np.ndarray:
        """Force per fibre on a fine excitation grid, shape ``(K, n_units)``."""
        grid = np.linspace(0.0, self.max_excitation, _TABLE_SIZE)
        return self.force_per_fiber(grid)

    def fast_force(self, fibers: np.ndarray, excitation: np.ndarray) -> np.ndarray:
        """Like :meth:`force`, using linear interpolation in a precomputed table.

        Excitation is clipped to ``[0, max_excitation]``. Recruitment makes
        per-unit force discontinuous, so the worst-case error is ~0.2% of
        healthy force (in narrow windows around thresholds); typical error
        is far smaller. About 3x faster than :meth:`force` in the engine.
        """
        table = self._force_table
        pos = np.clip(np.asarray(excitation) / self.max_excitation, 0.0, 1.0) * (_TABLE_SIZE - 1)
        i0 = np.minimum(pos.astype(np.intp), _TABLE_SIZE - 2)
        w = pos - i0
        f0 = np.einsum("...n,...n->...", fibers, table[i0])
        f1 = np.einsum("...n,...n->...", fibers, table[i0 + 1])
        return (f0 + w * (f1 - f0)) / self.healthy_max_force

    def force_excitation_curve(self, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(excitation fraction, force fraction)`` for the healthy pool."""
        frac = np.linspace(0.0, 1.0, n_points)
        force = self.force(self.fibers, frac * self.max_excitation)
        return frac, force
