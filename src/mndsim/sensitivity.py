"""Parameter sweeps and global sensitivity analysis.

:func:`global_sensitivity` samples patient parameters by Latin hypercube
(log-uniform for rates), simulates all samples in one vectorised batch and
ranks parameter influence on each output with partial rank correlation
coefficients (PRCC; Marino et al., 2008, *J Theor Biol* 254:178-196).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from .config import DEFAULT_CONFIG, PhysiologyConfig
from .engine import simulate
from .patient import POSITIVE_PARAMETERS, Patient
from .phenotypes import TYPICAL
from .treatment import TreatmentLike

DEFAULT_OUTPUTS: tuple[str, ...] = (
    "symptom_onset_months",
    "mu_loss_at_symptom_onset",
    "alsfrs_slope_first_year",
    "survival_from_symptom_onset",
)


def sweep(
    parameter: str,
    values: Sequence[float],
    *,
    base: Patient = TYPICAL,
    treatment: TreatmentLike = None,
    months: float = 180.0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
    **kwargs: Any,
) -> pd.DataFrame:
    """One-at-a-time sweep: summary metrics as one parameter varies."""
    patients = [base.replace(name=f"{parameter}={v:g}", **{parameter: v}) for v in values]
    res = simulate(patients, treatment, months=months, config=config, **kwargs)
    df = res.summary()
    df.insert(0, parameter, list(values))
    return df


def prcc(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partial rank correlation of each column of ``X`` with ``y``.

    Returns ``(coefficients, two-sided p-values)``. Rows where ``y`` is NaN
    are dropped; ``+inf`` is allowed (ranked highest), which is how outcomes
    beyond the simulated horizon are treated.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    n, k = X.shape
    if n <= k + 2:
        return np.full(k, np.nan), np.full(k, np.nan)
    rx = np.column_stack([stats.rankdata(X[:, j]) for j in range(k)])
    ry = stats.rankdata(y)
    coef = np.empty(k)
    for j in range(k):
        others = np.column_stack([np.ones(n), np.delete(rx, j, axis=1)])
        beta_x, *_ = np.linalg.lstsq(others, rx[:, j], rcond=None)
        beta_y, *_ = np.linalg.lstsq(others, ry, rcond=None)
        res_x = rx[:, j] - others @ beta_x
        res_y = ry - others @ beta_y
        denom = np.sqrt(np.sum(res_x**2) * np.sum(res_y**2))
        coef[j] = np.sum(res_x * res_y) / denom if denom > 0 else np.nan
    dof = n - 2 - (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef * np.sqrt(dof / np.maximum(1.0 - coef**2, 1e-300))
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return coef, p


@dataclass(frozen=True, eq=False)
class SensitivityResult:
    """Samples, outputs and PRCC table of a global sensitivity analysis."""

    samples: pd.DataFrame
    outputs: pd.DataFrame
    prcc: pd.DataFrame

    def ranking(self, output: str) -> pd.DataFrame:
        """PRCC table for one output, sorted by absolute influence."""
        df = self.prcc[self.prcc.output == output].copy()
        return df.reindex(df.prcc.abs().sort_values(ascending=False).index)


def global_sensitivity(
    ranges: Mapping[str, tuple[float, float]] | None = None,
    *,
    base: Patient = TYPICAL,
    n_samples: int = 256,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    treatment: TreatmentLike = None,
    months: float = 240.0,
    seed: int | None = 0,
    config: PhysiologyConfig = DEFAULT_CONFIG,
    dt: float = 0.5,
) -> SensitivityResult:
    """Latin-hypercube sampling + PRCC over patient parameters.

    Parameters
    ----------
    ranges:
        ``{parameter: (low, high)}``; rate parameters are sampled
        log-uniformly. Defaults to a factor of 2 either side of ``base`` for
        every rate parameter.
    outputs:
        Columns of :meth:`SimulationResult.summary` to analyse. Outcomes
        that are not reached within ``months`` are ranked as largest.
    """
    if ranges is None:
        ranges = {}
        for name in POSITIVE_PARAMETERS:
            v = getattr(base, name)
            if v > 0:
                ranges[name] = (max(v / 2.0, 1.0) if name == "max_enlargement" else v / 2.0, v * 2.0)
    names = list(ranges)
    if not names:
        raise ValueError("no parameters to vary")
    for name in names:
        if name not in POSITIVE_PARAMETERS:
            raise ValueError(f"cannot vary {name!r}; choose from {POSITIVE_PARAMETERS}")
        lo, hi = ranges[name]
        if not 0 < lo < hi:
            raise ValueError(f"invalid range for {name}: {ranges[name]}")
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n_samples)
    lo = np.log([ranges[n][0] for n in names])
    hi = np.log([ranges[n][1] for n in names])
    values = np.exp(lo + unit * (hi - lo))
    samples = pd.DataFrame(values, columns=names)
    patients = [
        base.replace(name=f"lhs{k:04d}", **{n: float(values[k, j]) for j, n in enumerate(names)})
        for k in range(n_samples)
    ]
    res = simulate(patients, treatment, months=months, dt=dt, config=config, seed=seed)
    summary = res.summary()
    out = summary[list(outputs)].astype(float)
    rows = []
    for col in outputs:
        y = out[col].to_numpy(copy=True)
        if col in ("survival_from_symptom_onset", "event_months", "symptom_onset_months"):
            y = np.where(np.isnan(y), np.inf, y)
        coef, p = prcc(values, y)
        for j, name in enumerate(names):
            rows.append({"output": col, "parameter": name, "prcc": coef[j], "p_value": p[j]})
    return SensitivityResult(samples=samples, outputs=out, prcc=pd.DataFrame(rows))
