"""Matplotlib figures for simulation, cohort, trial and sensitivity results.

Every function returns the :class:`matplotlib.figure.Figure` it draws, so
figures can be saved (``fig.savefig(...)``) or embedded. Colours follow
entities, not plot order: a region is always the same colour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .metrics import kaplan_meier
from .regions import REGIONS, region_index

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .result import SimulationResult
    from .sensitivity import SensitivityResult

# Validated categorical palette (fixed slot order) and neutral ink.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_SERIES = "#52514e"  # baseline / untreated / control
DIVERGING = ("#2a78d6", "#e34948")  # positive, negative

REGION_COLORS = dict(zip(REGIONS, SERIES[: len(REGIONS)], strict=True))
REGION_LABELS = {
    "bulbar": "Bulbar",
    "respiratory": "Respiratory",
    "upper_limb": "Upper limb",
    "lower_limb": "Lower limb",
}
TYPE_COLORS = {"S": SERIES[0], "FR": SERIES[1], "FF": SERIES[2]}
STATE_COLORS = ("#e1e0d9", SERIES[1], SERIES[0])  # dead, denervating, healthy
PARAMETER_LABELS = {
    "lmn_rate": "LMN degeneration rate",
    "umn_rate": "UMN loss rate",
    "excitotoxicity": "Excitotoxicity",
    "spread_rate": "Spread rate",
    "sprouting_rate": "Sprouting rate",
    "max_enlargement": "Max unit enlargement",
}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting requires matplotlib: pip install matplotlib") from exc
    return plt


def style_axes(
    ax: Axes, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None
) -> Axes:
    """Apply the quiet house style to an axes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8.5, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    return ax


def _figure(nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (7.0, 4.0), **kwargs: Any):
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kwargs)
    return fig, axes


def _legend(ax: Axes, **kwargs: Any) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6, **kwargs)
    if leg is not None:
        for line in leg.get_lines():
            line.set_linewidth(2.0)


def _end_label(ax: Axes, x: np.ndarray, y: np.ndarray, text: str) -> None:
    ok = np.isfinite(y)
    if ok.any():
        i = np.flatnonzero(ok)[-1]
        ax.annotate(
            text,
            (x[i], y[i]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )


def _mark_time(ax: Axes, t: float, label: str) -> None:
    """Vertical marker with its label just above the plot frame."""
    lo, hi = sorted(ax.get_xlim())
    if np.isfinite(t) and lo <= t <= hi:
        ax.axvline(t, color=MUTED, linewidth=0.8, zorder=1)
        ax.annotate(
            label,
            (t, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
            annotation_clip=False,
        )


# ----------------------------------------------------------------------
# Single-patient views
# ----------------------------------------------------------------------


def plot_result(result: SimulationResult, patient: int = 0, *, aligned: bool = False) -> Figure:
    """Six-panel overview of one patient's simulated disease course.

    Panels: regional strength; motor-unit number vs CMAP vs strength in the
    region of onset (compensation); survival by unit type; functional
    score; respiratory capacity; size of surviving units.
    """
    p = patient
    patient_obj = result.patients[p]
    shift = result.symptom_onset[p] if aligned and np.isfinite(result.symptom_onset[p]) else 0.0
    t = result.time - shift
    xlabel = "Months since symptom onset" if shift else "Months since disease onset"
    onset_r = region_index(patient_obj.onset_region)
    onset_name = REGION_LABELS[patient_obj.onset_region].lower()
    events = [(result.symptom_onset[p] - shift, "symptoms"), (result.event_time[p] - shift, "resp. failure")]

    fig, axes = _figure(2, 3, size=(13.0, 7.2), constrained_layout=True)
    ax = style_axes(axes[0, 0], title="Strength by region", ylabel="Max voluntary force (% healthy)")
    for r, region in enumerate(REGIONS):
        y = 100 * result.mvc[p, :, r]
        ax.plot(t, y, color=REGION_COLORS[region], lw=2, label=REGION_LABELS[region])
    ax.axhline(100 * result.config.symptom_threshold, color=AXIS, lw=0.8, zorder=1)
    ax.set_ylim(0, 105)
    _legend(ax, loc="lower left")

    ax = style_axes(axes[0, 1], title=f"Compensation in the {onset_name}", ylabel="% of baseline")
    series = (
        ("Motor unit number", 100 * result.mu_count[p, :, onset_r] / result.pool.n_units, SERIES[0]),
        ("CMAP amplitude", 100 * result.cmap[p, :, onset_r], SERIES[1]),
        ("Strength", 100 * result.mvc[p, :, onset_r], SERIES[2]),
    )
    for label, y, color in series:
        ax.plot(t, y, color=color, lw=2, label=label)
    ax.set_ylim(0, 105)
    _legend(ax, loc="lower left")

    ax = style_axes(axes[0, 2], title=f"Surviving units by type ({onset_name})", ylabel="% surviving")
    for k, name in enumerate(("S", "FR", "FF")):
        ax.plot(
            t,
            100 * result.type_survival[p, :, onset_r, k],
            color=TYPE_COLORS[name],
            lw=2,
            label={"S": "Slow (S)", "FR": "Fast resistant (FR)", "FF": "Fast fatigable (FF)"}[name],
        )
    ax.set_ylim(0, 105)
    _legend(ax, loc="upper right")

    ax = style_axes(
        axes[1, 0], title="Functional score (ALSFRS-R-like)", xlabel=xlabel, ylabel="Score (0-48)"
    )
    ax.plot(t, result.alsfrs[p], color=INK, lw=2)
    ax.set_ylim(0, 49)

    ax = style_axes(axes[1, 1], title="Respiratory capacity", xlabel=xlabel, ylabel="FVC (% predicted)")
    ax.plot(t, result.fvc[p], color=INK, lw=2)
    ax.axhline(100 * result.config.respiratory_failure_threshold, color=AXIS, lw=0.8, zorder=1)
    ax.set_ylim(0, 105)

    ax = style_axes(
        axes[1, 2],
        title=f"Surviving-unit size ({onset_name})",
        xlabel=xlabel,
        ylabel="Fibres per unit (x original)",
    )
    ax.plot(t, result.enlargement[p, :, onset_r], color=INK, lw=2)
    ax.set_ylim(bottom=0.9)

    for ax in axes.flat:
        ax.set_xlim(t[0], t[-1])
        for when, label in events:
            _mark_time(ax, when, label)
    tx = ", ".join(x.name for x in result.treatment) or "untreated"
    fig.suptitle(f"{patient_obj.name} - {tx}", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    return fig


def plot_motor_units(result: SimulationResult, patient: int = 0, region: str | None = None) -> Figure:
    """Raster of every motor unit's state over time (requires ``record_units=True``).

    Units are ordered by size (Henneman's principle) from bottom to top. In
    mean-field mode colours blend in proportion to state probabilities.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    if result.units_healthy is None:
        raise ValueError("simulate(..., record_units=True) is required for this plot")
    region = region or result.patients[patient].onset_region
    r = region_index(region)
    healthy = result.units_healthy[patient, :, r].astype(float)
    denerv = result.units_denervating[patient, :, r].astype(float)
    state = 2.0 * healthy + 1.0 * denerv  # 0 dead, 1 denervating, 2 healthy
    cmap = LinearSegmentedColormap.from_list("unit_state", STATE_COLORS)

    fig, ax = _figure(size=(9.0, 4.2), constrained_layout=True)
    style_axes(
        ax,
        title=f"Motor-unit fates, {REGION_LABELS[region].lower()} pool",
        xlabel="Months since disease onset",
        ylabel="Motor unit (small to large)",
    )
    ax.grid(False)
    n = result.pool.n_units
    ax.imshow(
        state.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=(result.time[0], result.time[-1], 0.5, n + 0.5),
        interpolation="nearest",
    )
    types = result.pool.unit_type
    for k, name in enumerate(("S", "FR", "FF")):
        idx = np.flatnonzero(types == k)
        if idx.size:
            if k > 0:
                ax.axhline(idx[0] + 0.5, color=INK_2, lw=0.6)
            ax.annotate(
                name,
                (1.0, (idx[0] + idx[-1]) / 2 + 1),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_2,
            )
    handles = [
        Patch(color=c, label=lab)
        for c, lab in zip(STATE_COLORS[::-1], ("Healthy", "Denervating", "Lost"), strict=True)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    _mark_time(ax, result.symptom_onset[patient], "symptoms")
    return fig


# ----------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------


def plot_trajectories(
    results: Mapping[str, SimulationResult],
    metric: str = "alsfrs",
    *,
    patient: int = 0,
    region: str | None = None,
    aligned: bool = True,
    marks: Mapping[str, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Overlay one metric across several simulations (e.g. counterfactual treatments).

    The first entry is drawn in neutral ink as the reference. ``marks``
    adds labelled vertical lines (e.g. treatment start times, on the same
    time axis as the plot).
    """
    labels = {
        "alsfrs": "Functional score (0-48)",
        "fvc": "FVC (% predicted)",
        "mvc": "Strength (% healthy)",
        "mu_count": "Motor units",
    }
    fig, ax = _figure(size=(8.0, 4.4), constrained_layout=True)
    ref = next(iter(results.values()))
    shift = ref.symptom_onset[patient] if aligned and np.isfinite(ref.symptom_onset[patient]) else 0.0
    style_axes(
        ax,
        title=title or labels.get(metric, metric),
        xlabel="Months since symptom onset" if shift else "Months since disease onset",
        ylabel=labels.get(metric, metric),
    )
    for k, (label, res) in enumerate(results.items()):
        y = getattr(res, metric)[patient]
        if y.ndim == 2:
            y = y[:, region_index(region or res.patients[patient].onset_region)]
        if metric == "mvc":
            y = 100 * y
        color = NEUTRAL_SERIES if k == 0 else SERIES[(k - 1) % len(SERIES)]
        ax.plot(res.time - shift, y, color=color, lw=2.4 if k == 0 else 2, label=label)
    ax.set_xlim(ref.time[0] - shift, ref.time[-1] - shift)
    for label, when in (marks or {}).items():
        _mark_time(ax, when, label)
    if metric == "alsfrs":
        ax.set_ylim(0, 49)
    _legend(ax, loc="lower left")
    return fig


def plot_compensation(results: Mapping[str, SimulationResult], patient: int = 0) -> Figure:
    """Strength against motor-unit loss in the region of onset.

    Shows the "clinical threshold": reinnervation hides loss of a large
    fraction of motor units before strength falls.
    """
    fig, ax = _figure(size=(6.4, 4.4), constrained_layout=True)
    style_axes(
        ax,
        title="Strength is preserved until compensation is exhausted",
        xlabel="Motor units lost in region of onset (%)",
        ylabel="Strength (% healthy)",
    )
    for k, (label, res) in enumerate(results.items()):
        r = region_index(res.patients[patient].onset_region)
        lost = 100 * res.mu_loss[patient, :, r]
        ax.plot(lost, 100 * res.mvc[patient, :, r], color=SERIES[k % len(SERIES)], lw=2, label=label)
    ax.plot([0, 100], [100, 0], color=AXIS, lw=0.9, zorder=1)
    ax.annotate(
        "strength proportional to unit count",
        (52, 49),
        fontsize=7.5,
        color=MUTED,
        rotation=-33,
        ha="center",
        va="bottom",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    _legend(ax, loc="lower left")
    return fig


def plot_cohort(
    result: SimulationResult,
    metric: str = "alsfrs",
    *,
    aligned: bool = True,
    max_lines: int = 150,
    months: float = 60.0,
) -> Figure:
    """Individual trajectories (thin) and per-onset-region medians (bold).

    Trajectories continue after the respiratory-failure event (the composite
    includes permanent ventilation), so medians are free of survivor bias.
    """
    fig, ax = _figure(size=(8.0, 4.6), constrained_layout=True)
    style_axes(
        ax,
        title="Virtual cohort: decline by region of onset",
        xlabel="Months since symptom onset" if aligned else "Months",
        ylabel="Functional score (0-48)" if metric == "alsfrs" else metric,
    )
    grid = np.arange(0.0, months + 0.5, 1.0)
    data = getattr(result, metric)
    curves = np.full((result.n_patients, grid.size), np.nan)
    for p in range(result.n_patients):
        shift = result.symptom_onset[p] if aligned else 0.0
        if np.isfinite(shift):
            curves[p] = np.interp(grid, result.time - shift, data[p], left=np.nan, right=np.nan)
    onset = np.array([p.onset_region for p in result.patients])
    rng = np.random.default_rng(0)
    show = rng.permutation(result.n_patients)[:max_lines]
    for p in show:
        ax.plot(grid, curves[p], color=REGION_COLORS[onset[p]], lw=0.6, alpha=0.25)
    for region in REGIONS:
        sel = onset == region
        if sel.sum() >= 5:
            med = _median_while_enough(curves[sel])
            ax.plot(
                grid,
                med,
                color=REGION_COLORS[region],
                lw=2.4,
                label=f"{REGION_LABELS[region]} onset (n={sel.sum()})",
            )
    ax.set_xlim(0, months)
    _legend(ax, loc="lower left")
    return fig


def _median_while_enough(curves: np.ndarray, min_frac: float = 0.5) -> np.ndarray:
    """Median across rows, shown only while at least ``min_frac`` are still followed."""
    import warnings

    n_alive = np.sum(np.isfinite(curves), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        med = np.nanmedian(curves, axis=0)
    return np.where(n_alive >= min_frac * curves.shape[0], med, np.nan)


def plot_kaplan_meier(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str = "Survival",
    xlabel: str = "Months",
) -> Figure:
    """Kaplan-Meier curves; ``groups`` maps label -> (times, event_observed)."""
    fig, ax = _figure(size=(6.8, 4.2), constrained_layout=True)
    style_axes(ax, title=title, xlabel=xlabel, ylabel="Event-free (%)")
    for k, (label, (times, events)) in enumerate(groups.items()):
        t, s = kaplan_meier(times, events)
        color = NEUTRAL_SERIES if k == 0 and len(groups) > 1 else SERIES[max(k - 1, 0) % len(SERIES)]
        ax.step(t, 100 * s, where="post", color=color, lw=2, label=label)
    ax.set_ylim(0, 102)
    _legend(ax, loc="lower left")
    return fig


def plot_power_curve(df: pd.DataFrame, *, target: float = 0.8, title: str | None = None) -> Figure:
    """Power against sample size for each endpoint (output of :func:`mndsim.trial.power_curve`)."""
    names = {
        "alsfrs_slope": "Functional slope",
        "survival": "Survival (log-rank)",
        "cafs": "CAFS (function + survival)",
        "mune_slope": "Motor unit number",
        "fvc_slope": "FVC slope",
        "nfl": "Plasma NfL",
    }
    fig, ax = _figure(size=(8.6, 4.4), constrained_layout=True)
    style_axes(ax, title=title or "Power by endpoint", xlabel="Patients per arm", ylabel="Power (%)")
    for k, (ep, d) in enumerate(df.groupby("endpoint", sort=False)):
        ax.plot(
            d.n_per_arm,
            100 * d.power,
            color=SERIES[k % len(SERIES)],
            lw=2,
            marker="o",
            ms=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=names.get(ep, ep),
        )
    ax.axhline(100 * target, color=AXIS, lw=0.9, zorder=1)
    ax.set_ylim(0, 102)
    _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_sensitivity(result: SensitivityResult, output: str, *, title: str | None = None) -> Figure:
    """Tornado chart of partial rank correlation coefficients for one output."""
    df = result.ranking(output).iloc[::-1]
    fig, ax = _figure(size=(6.4, 0.45 * len(df) + 1.4), constrained_layout=True)
    style_axes(
        ax, title=title or f"What drives {output.replace('_', ' ')}?", xlabel="Partial rank correlation"
    )
    colors = [DIVERGING[0] if v >= 0 else DIVERGING[1] for v in df.prcc]
    ax.barh([PARAMETER_LABELS.get(n, n) for n in df.parameter], df.prcc, color=colors, height=0.55)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(df.prcc):
        ax.annotate(
            f"{v:+.2f}",
            (v, y),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    return fig


def plot_hidden_phase(result: SimulationResult, patient: int = 0) -> Figure:
    """Motor-unit loss, strength, function and blood NfL around symptom onset.

    Two panels share a time axis (months from first symptom): the hidden
    phase is shaded. NfL is shown as a multiple of its healthy level.
    """
    p = patient
    onset = result.symptom_onset[p]
    if not np.isfinite(onset):
        raise ValueError("patient never becomes symptomatic within the horizon")
    pat = result.patients[p]
    r = region_index(pat.onset_region)
    t = result.time - onset
    region = REGION_LABELS[pat.onset_region].lower()
    fig, axes = _figure(1, 2, size=(12.0, 4.2), constrained_layout=True)
    ax = style_axes(
        axes[0],
        title=f"Loss is hidden until compensation runs out ({region})",
        xlabel="Months from first symptom",
        ylabel="% of healthy",
    )
    ax.plot(
        t,
        100 * result.mu_count[p, :, r] / result.pool.n_units,
        color=SERIES[0],
        lw=2,
        label="Motor neurons alive",
    )
    ax.plot(t, 100 * result.mvc[p, :, r], color=SERIES[1], lw=2, label="Strength")
    ax.plot(t, 100 * result.alsfrs[p] / result.max_score, color=SERIES[2], lw=2, label="Daily-function score")
    ax.set_ylim(0, 105)
    _legend(ax, loc="upper right")
    ax2 = style_axes(
        axes[1],
        title="Blood NfL reports the rate of loss",
        xlabel="Months from first symptom",
        ylabel="NfL (x healthy for age)",
    )
    ax2.plot(t, result.nfl[p] / result.nfl[p, 0], color=SERIES[4], lw=2)
    ax2.axhline(2.0, color=AXIS, lw=0.9, zorder=1)
    ax2.annotate(
        "doubled",
        (t[-1], 2.0),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    ax2.set_ylim(bottom=0)
    lo, hi = t[0], min(t[-1], 60.0)
    for a in (ax, ax2):
        a.set_xlim(lo, hi)
        a.axvspan(lo, 0, color=SERIES[0], alpha=0.07, lw=0)
        a.annotate(
            "hidden phase",
            (lo, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
        _mark_time(a, 0.0, "symptoms")
        _mark_time(a, pat.diagnostic_delay, "diagnosis")
        _mark_time(a, result.event_time[p] - onset, "resp. failure")
    return fig


def plot_power_by_duration(
    panels: Mapping[str, pd.DataFrame],
    endpoints: Sequence[str] = ("nfl", "alsfrs_slope"),
    *,
    title: str | None = None,
) -> Figure:
    """Small multiples of power against trial length (one panel per treatment).

    ``panels`` maps a panel title to the output of
    :func:`mndsim.trial.power_by_duration`.
    """
    names = {
        "alsfrs_slope": "Functional slope",
        "survival": "Survival",
        "cafs": "CAFS",
        "mune_slope": "Motor unit number",
        "fvc_slope": "FVC slope",
        "nfl": "Blood NfL",
    }
    colours = {
        "nfl": SERIES[4],
        "alsfrs_slope": SERIES[0],
        "fvc_slope": SERIES[3],
        "survival": SERIES[1],
        "mune_slope": SERIES[2],
        "cafs": SERIES[6],
    }
    fig, axes = _figure(1, len(panels), size=(5.2 * len(panels), 4.0), constrained_layout=True, squeeze=False)
    for ax, (label, df) in zip(axes[0], panels.items(), strict=True):
        style_axes(ax, title=label, xlabel="Trial length (months)", ylabel="Power (%)")
        for ep in endpoints:
            d = df[df.endpoint == ep]
            ax.plot(
                d.duration,
                100 * d.power,
                color=colours.get(ep, INK),
                lw=2,
                marker="o",
                ms=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=names.get(ep, ep),
            )
        ax.axhline(80, color=AXIS, lw=0.9, zorder=1)
        ax.set_ylim(0, 102)
        _legend(ax, loc="best")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    return fig


def plot_prevention(strategies: pd.DataFrame, tradeoff: pd.DataFrame) -> Figure:
    """NfL-triggered treatment: outcomes by strategy, and the threshold trade-off.

    ``strategies`` is :meth:`mndsim.hidden_phase.StrategyResult.summary`;
    ``tradeoff`` is :func:`mndsim.hidden_phase.threshold_tradeoff`.
    """
    labels = {
        "untreated": "Untreated",
        "diagnosis": "Start at diagnosis",
        "symptom_onset": "Start at first symptom",
        "nfl_trigger": "Start when NfL rises",
    }
    fig, axes = _figure(1, 2, size=(12.0, 4.2), constrained_layout=True)
    ax = style_axes(
        axes[0],
        title="Function two years after the first symptom",
        xlabel="Median daily-function score (0-48)",
    )
    order = [k for k in ("untreated", "diagnosis", "symptom_onset", "nfl_trigger") if k in strategies.index]
    vals = [strategies.loc[k, "median_function_at_+24mo"] for k in order]
    colours = [NEUTRAL_SERIES, SERIES[1], SERIES[0], SERIES[4]][: len(order)]
    ax.barh([labels[k] for k in order], vals, color=colours, height=0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 48)
    for y, v in enumerate(vals):
        ax.annotate(
            f"{v:.0f}",
            (v, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax2 = style_axes(
        axes[1],
        title="Choosing the NfL threshold",
        xlabel="False alarms in non-converting carriers over 10 years (%)",
        ylabel="Carriers treated before symptoms (%)",
    )
    ax2.plot(
        100 * tradeoff.false_trigger_rate_10y,
        100 * tradeoff.triggered_before_symptoms,
        color=SERIES[4],
        lw=2,
        marker="o",
        ms=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    for _, row in tradeoff.iterrows():
        ax2.annotate(
            f"{row.threshold:g}x",
            (100 * row.false_trigger_rate_10y, 100 * row.triggered_before_symptoms),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    ax2.set_xlim(-2, max(5.0, 100 * tradeoff.false_trigger_rate_10y.max() * 1.1))
    ax2.set_ylim(0, 100)
    return fig


def save(fig: Figure, path: str, dpi: int = 160) -> None:
    """Save with the house surface colour."""
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")


__all__: Sequence[str] = (
    "plot_cohort",
    "plot_compensation",
    "plot_kaplan_meier",
    "plot_motor_units",
    "plot_power_curve",
    "plot_result",
    "plot_sensitivity",
    "plot_trajectories",
    "save",
    "style_axes",
)
