"""Clinical read-outs derived from regional motor capacity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .regions import FUNCTIONAL_ITEMS, REGIONS, FunctionalItem


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def item_scores(capacity: np.ndarray, items: Sequence[FunctionalItem] = FUNCTIONAL_ITEMS) -> np.ndarray:
    """Expected item scores (0-4) from regional capacity.

    ``capacity`` has shape ``(..., n_regions)`` (1 = healthy strength). The
    result has shape ``(..., n_items)``. Each item follows a logistic curve
    rescaled so that capacity 0 maps exactly to 0 and capacity 1 to 4.
    """
    capacity = np.asarray(capacity, dtype=float)
    out = np.empty((*capacity.shape[:-1], len(items)))
    for k, item in enumerate(items):
        idx = [REGIONS.index(r) for r in item.regions]
        cap = capacity[..., idx].mean(axis=-1)
        lo = _logistic(np.asarray(-item.c50 / item.width))
        hi = _logistic(np.asarray((1.0 - item.c50) / item.width))
        val = (_logistic((cap - item.c50) / item.width) - lo) / (hi - lo)
        out[..., k] = 4.0 * np.clip(val, 0.0, 1.0)
    return out


def observe_items(expected: np.ndarray, rng: np.random.Generator, noise_sd: float = 0.3) -> np.ndarray:
    """Turn expected item scores into integer clinician ratings.

    Adds Gaussian rater noise (SD ``noise_sd`` item points), then rounds and
    clips to the 0-4 ordinal scale.
    """
    noisy = expected + rng.normal(0.0, noise_sd, size=np.shape(expected))
    return np.clip(np.rint(noisy), 0, 4)


def fvc_percent(respiratory_capacity: np.ndarray) -> np.ndarray:
    """Forced-vital-capacity proxy, % predicted, from respiratory pool strength."""
    return 100.0 * np.clip(respiratory_capacity, 0.0, None)
