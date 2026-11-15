"""Fit a virtual patient ("digital twin") to longitudinal observations.

Given a patient's functional scores (and optionally FVC and motor unit
number estimates) at known months since symptom onset, :func:`fit_patient`
finds disease parameters whose simulated trajectory matches the data,
using a coarse vectorised grid search followed by bounded non-linear
least squares in log-parameter space. The fitted :class:`Patient` can then
be used to explore counterfactual treatment scenarios.

Identifiability is limited: a functional score alone mostly constrains the
overall speed of loss and of spread. Fit few parameters, prefer adding FVC
or MUNE data, and read the reported standard errors.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from .config import DEFAULT_CONFIG, PhysiologyConfig
from .engine import simulate
from .patient import POSITIVE_PARAMETERS, Patient
from .phenotypes import TYPICAL
from .regions import region_index
from .result import SimulationResult
from .treatment import TreatmentLike

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "lmn_rate": (0.005, 0.6),
    "umn_rate": (0.0005, 0.2),
    "excitotoxicity": (0.01, 3.0),
    "spread_rate": (0.002, 0.5),
    "sprouting_rate": (0.1, 10.0),
    "max_enlargement": (1.2, 12.0),
}

_PENALTY = 25.0  # residual (in SDs) assigned to points the model cannot reach


@dataclass(frozen=True, eq=False)
class FitResult:
    """Outcome of :func:`fit_patient`."""

    patient: Patient
    parameters: dict[str, float]
    log_se: dict[str, float]
    rmse_alsfrs: float
    n_observations: int
    success: bool
    message: str
    observations: pd.DataFrame
    symptom_onset: float
    config: PhysiologyConfig

    def confidence_intervals(self, level: float = 0.95) -> pd.DataFrame:
        """Approximate (log-normal) confidence intervals from the Jacobian."""
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2)
        rows = []
        for name, value in self.parameters.items():
            se = self.log_se.get(name, float("nan"))
            rows.append(
                {
                    "parameter": name,
                    "estimate": value,
                    "lower": value * math.exp(-z * se),
                    "upper": value * math.exp(z * se),
                    "log_se": se,
                }
            )
        return pd.DataFrame(rows)

    def simulate(
        self,
        treatment: TreatmentLike = None,
        *,
        months_after_onset: float = 60.0,
        **kwargs,
    ) -> SimulationResult:
        """Simulate the fitted patient (optionally treated) from disease onset."""
        return simulate(
            self.patient,
            treatment,
            months=self.symptom_onset + months_after_onset,
            config=self.config,
            **kwargs,
        )

    def counterfactual(
        self, treatment: TreatmentLike, start_after_onset: float, months_after_onset: float = 60.0
    ) -> SimulationResult:
        """Simulate the fitted patient with a treatment started at a given
        number of months after *symptom* onset."""
        from .treatment import Combination, Treatment, as_treatment_list

        shifted: list[Treatment] = [
            t.at(self.symptom_onset + start_after_onset, t.stop) for t in as_treatment_list(treatment)
        ]
        return self.simulate(Combination(tuple(shifted)), months_after_onset=months_after_onset)


def _residuals(
    result: SimulationResult,
    obs_t: np.ndarray,
    alsfrs: np.ndarray | None,
    fvc: np.ndarray | None,
    mune: np.ndarray | None,
    mune_region: int,
    sd: Mapping[str, float],
) -> np.ndarray:
    """Standardised residuals for every patient in ``result`` -> (P, n_res)."""
    P = result.n_patients
    out = []
    onset = result.symptom_onset
    for p in range(P):
        if not np.isfinite(onset[p]):
            n = sum(x.size for x in (alsfrs, fvc, mune) if x is not None)
            out.append(np.full(n, _PENALTY))
            continue
        t = onset[p] + obs_t
        parts = []
        if alsfrs is not None:
            pred = np.interp(t, result.time, result.alsfrs[p], right=np.nan)
            parts.append((pred - alsfrs) / sd["alsfrs"])
        if fvc is not None:
            pred = np.interp(t, result.time, result.fvc[p], right=np.nan)
            parts.append((pred - fvc) / sd["fvc"])
        if mune is not None:
            pred = np.interp(t, result.time, result.mu_count[p, :, mune_region], right=np.nan)
            parts.append((np.log(np.maximum(pred, 0.5)) - np.log(np.maximum(mune, 0.5))) / sd["log_mune"])
        r = np.concatenate(parts)
        out.append(np.where(np.isfinite(r), r, _PENALTY))
    return np.array(out)


def fit_patient(
    months_since_onset: Sequence[float],
    alsfrs: Sequence[float] | None = None,
    *,
    fvc: Sequence[float] | None = None,
    mune: Sequence[float] | None = None,
    onset_region: str = "upper_limb",
    base: Patient = TYPICAL,
    parameters: Sequence[str] = ("lmn_rate", "spread_rate"),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    grid_points: int = 9,
    noise_sd: Mapping[str, float] | None = None,
    config: PhysiologyConfig = DEFAULT_CONFIG,
    dt: float = 0.5,
    horizon: float = 240.0,
) -> FitResult:
    """Fit disease parameters to one patient's longitudinal measurements.

    Parameters
    ----------
    months_since_onset:
        Assessment times, in months since *symptom* onset.
    alsfrs, fvc, mune:
        Observed functional score (0-48), FVC (% predicted) and motor unit
        number estimate (same scale as the model's pool, 120 when healthy)
        at those times. At least one is required; NaN entries are ignored.
    onset_region:
        Region where symptoms began.
    base:
        Patient supplying the parameters that are *not* fitted.
    parameters:
        Names of the :class:`Patient` rate parameters to fit.
    bounds:
        Optional ``{name: (low, high)}`` overrides of :data:`DEFAULT_BOUNDS`.
    grid_points:
        Grid resolution per parameter for the initial search.
    noise_sd:
        Measurement SDs used to weight residuals; defaults
        ``{"alsfrs": 2.0, "fvc": 8.0, "log_mune": 0.2}``.
    """
    t = np.asarray(months_since_onset, dtype=float)
    series = {}
    for name, values in (("alsfrs", alsfrs), ("fvc", fvc), ("mune", mune)):
        if values is not None:
            v = np.asarray(values, dtype=float)
            if v.shape != t.shape:
                raise ValueError(f"{name} must have the same length as months_since_onset")
            series[name] = v
    if not series:
        raise ValueError("provide at least one of alsfrs, fvc or mune")
    keep = np.ones(t.shape, dtype=bool)
    for v in series.values():
        keep &= np.isfinite(v)
    keep &= np.isfinite(t) & (t >= 0)
    if keep.sum() < 2:
        raise ValueError("need at least two complete observations")
    t = t[keep]
    series = {k: v[keep] for k, v in series.items()}
    for name in parameters:
        if name not in POSITIVE_PARAMETERS:
            raise ValueError(f"cannot fit {name!r}; choose from {POSITIVE_PARAMETERS}")
    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    lo = np.log([b[n][0] for n in parameters])
    hi = np.log([b[n][1] for n in parameters])
    sd = {"alsfrs": 2.0, "fvc": 8.0, "log_mune": 0.2, **(noise_sd or {})}
    template = base.replace(onset_region=onset_region, seed=base.seed if base.seed is not None else 0)
    mune_r = region_index(onset_region)
    stop_after = float(t.max()) + 1.0

    def make(theta: np.ndarray) -> Patient:
        return template.replace(**{n: float(math.exp(v)) for n, v in zip(parameters, theta, strict=True)})

    def residual_matrix(thetas: np.ndarray) -> np.ndarray:
        res = simulate(
            [make(th) for th in thetas],
            months=horizon,
            dt=dt,
            record_every=dt,
            config=config,
            stop_after_onset=stop_after,
        )
        return _residuals(res, t, series.get("alsfrs"), series.get("fvc"), series.get("mune"), mune_r, sd)

    # 1. Coarse grid search (vectorised).
    axes = [np.linspace(lo[k], hi[k], grid_points) for k in range(len(parameters))]
    grid = np.array(list(itertools.product(*axes)))
    sse = np.sum(residual_matrix(grid) ** 2, axis=1)
    starts = grid[np.argsort(sse)[:3]]

    # 2. Local refinement from the best grid points.
    best = None
    for start in starts:
        sol = optimize.least_squares(
            lambda th: residual_matrix(th[None, :])[0],
            start,
            bounds=(lo, hi),
            diff_step=1e-3,
            x_scale=1.0,
            max_nfev=200,
        )
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None

    n_res = best.fun.size
    dof = max(n_res - len(parameters), 1)
    s2 = 2.0 * best.cost / dof
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:  # pragma: no cover
        se = np.full(len(parameters), np.nan)

    patient = make(best.x).replace(name=f"fitted_{base.name}")
    fitted = simulate(
        patient, months=horizon, dt=dt, record_every=dt, config=config, stop_after_onset=stop_after
    )
    onset = float(fitted.symptom_onset[0])
    obs = pd.DataFrame({"months_since_onset": t, **{f"observed_{k}": v for k, v in series.items()}})
    if np.isfinite(onset):
        obs["predicted_alsfrs"] = np.interp(onset + t, fitted.time, fitted.alsfrs[0])
        obs["predicted_fvc"] = np.interp(onset + t, fitted.time, fitted.fvc[0])
        obs["predicted_mune"] = np.interp(onset + t, fitted.time, fitted.mu_count[0, :, mune_r])
    rmse = (
        float(np.sqrt(np.mean((obs["predicted_alsfrs"] - obs["observed_alsfrs"]) ** 2)))
        if "observed_alsfrs" in obs and "predicted_alsfrs" in obs
        else float("nan")
    )
    return FitResult(
        patient=patient,
        parameters={n: float(math.exp(v)) for n, v in zip(parameters, best.x, strict=True)},
        log_se={n: float(s) for n, s in zip(parameters, se, strict=True)},
        rmse_alsfrs=rmse,
        n_observations=int(t.size),
        success=bool(best.success),
        message=str(best.message),
        observations=obs,
        symptom_onset=onset,
        config=config,
    )
