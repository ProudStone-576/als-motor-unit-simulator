"""Small, dependency-free trajectory and survival metrics."""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201 (NumPy < 2.0)


def interpolate_at(t: np.ndarray, y: np.ndarray, at: float) -> float:
    """Linear interpolation of ``y(t)`` at ``at``; NaN outside the range or if ``at`` is NaN."""
    if not np.isfinite(at) or at < t[0] or at > t[-1]:
        return float("nan")
    return float(np.interp(at, t, y))


def time_to_threshold(t: np.ndarray, y: np.ndarray, threshold: float, direction: str = "below") -> float:
    """First time ``y`` crosses ``threshold``, linearly interpolated.

    Returns NaN if the threshold is never crossed. ``direction`` is
    ``"below"`` (default) or ``"above"``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if direction == "above":
        y, threshold = -y, -threshold
    elif direction != "below":
        raise ValueError("direction must be 'below' or 'above'")
    below = np.flatnonzero(y < threshold)
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    frac = (y0 - threshold) / (y0 - y1) if y0 != y1 else 0.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def auc(t: np.ndarray, y: np.ndarray) -> float:
    """Area under the curve by the trapezoidal rule."""
    return float(_trapezoid(np.asarray(y, dtype=float), np.asarray(t, dtype=float)))


def slope(t: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least-squares slope of ``y`` on ``t`` (ignores NaNs)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    tc = t[ok] - t[ok].mean()
    denom = np.sum(tc**2)
    return float(np.sum(tc * (y[ok] - y[ok].mean())) / denom) if denom > 0 else float("nan")


def slopes(t: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Row-wise OLS slopes for a matrix ``y`` (patients x visits).

    ``t`` may be 1-D (shared visit times) or 2-D. Entries where ``mask`` is
    False (or ``y`` is NaN) are ignored. Rows with fewer than two valid
    points give NaN.
    """
    y = np.asarray(y, dtype=float)
    t = np.broadcast_to(np.asarray(t, dtype=float), y.shape)
    ok = np.isfinite(y) & np.isfinite(t)
    if mask is not None:
        ok &= mask
    n = ok.sum(-1)
    w = ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.sum(w * np.where(ok, t, 0), -1) / n
        ym = np.sum(w * np.where(ok, y, 0), -1) / n
        tc = np.where(ok, t - tm[:, None], 0.0)
        yc = np.where(ok, y - ym[:, None], 0.0)
        num = np.sum(tc * yc, -1)
        den = np.sum(tc**2, -1)
        out = num / den
    return np.where((n >= 2) & (den > 0), out, np.nan)


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve.

    Parameters
    ----------
    times:
        Event or censoring times.
    events:
        True where the event was observed, False where censored.

    Returns
    -------
    (t, survival)
        Step-function knots, starting at ``(0, 1)``.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    unique = np.unique(times[events])
    s = 1.0
    out_t, out_s = [0.0], [1.0]
    for u in unique:
        at_risk = np.sum(times >= u)
        d = np.sum((times == u) & events)
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        out_t.append(float(u))
        out_s.append(s)
    return np.array(out_t), np.array(out_s)


def median_survival(times: np.ndarray, events: np.ndarray) -> float:
    """Kaplan-Meier median survival (NaN if survival never drops to 0.5)."""
    t, s = kaplan_meier(times, events)
    below = np.flatnonzero(s <= 0.5)
    return float(t[below[0]]) if below.size else float("nan")
