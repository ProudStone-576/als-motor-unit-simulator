"""Command-line interface: ``mndsim <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import __version__
from .patient import Patient
from .phenotypes import PHENOTYPES, get_phenotype
from .treatment import TREATMENTS, Combination, Treatment, get_treatment


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _patient(args: argparse.Namespace) -> Patient:
    if getattr(args, "patient_json", None):
        with open(args.patient_json, encoding="utf-8") as fh:
            patient = Patient.from_dict(json.load(fh))
    else:
        patient = get_phenotype(args.phenotype)
    changes = {}
    for item in getattr(args, "param", None) or []:
        if "=" not in item:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        key, value = item.split("=", 1)
        changes[key.strip()] = _parse_value(value.strip())
    return patient.replace(**changes) if changes else patient


def _treatments(names: Sequence[str] | None) -> list[Treatment]:
    return [get_treatment(n) for n in names or []]


def _save_fig(fig, path: str | None) -> None:
    if path:
        from .plotting import save

        save(fig, path)
        print(f"figure written to {path}", file=sys.stderr)


def _print_frame(df, fmt: str = "{:.3g}") -> None:
    import pandas as pd

    with pd.option_context("display.width", 160, "display.max_columns", 50):
        print(df.to_string(index=False, float_format=lambda v: fmt.format(v)))


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


# ----------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    if args.what in ("phenotypes", "all"):
        print("Phenotypes:")
        for name, p in PHENOTYPES.items():
            print(
                f"  {name:18s} onset={p.onset_region:11s} lmn={p.lmn_rate:<6g} umn={p.umn_rate:<6g} "
                f"exc={p.excitotoxicity:<4g} spread={p.spread_rate:<6g} genotype={p.genotype}"
            )
    if args.what in ("treatments", "all"):
        print("Treatments:")
        for name, t in TREATMENTS.items():
            print(f"  {name:18s} {t.description}")
    return 0


def _resolve_start(patient: Patient, args: argparse.Namespace, months: float) -> tuple[float, str]:
    """Return (start in months since disease onset, human description)."""
    if args.start is not None:
        return float(args.start), f"month {args.start:g} after disease onset"
    from .engine import simulate

    after = patient.diagnostic_delay if args.start_after_onset is None else args.start_after_onset
    onset = simulate(patient, months=months, dt=0.5).symptom_onset[0]
    if not np.isfinite(onset):
        raise SystemExit("this patient never becomes symptomatic; use --start instead")
    return float(onset + after), f"{after:g} months after symptom onset"


def cmd_simulate(args: argparse.Namespace) -> int:
    from .engine import simulate

    _use_agg()
    patient = _patient(args)
    txs = _treatments(args.treatment)
    if txs:
        start, how = _resolve_start(patient, args, args.months)
        txs = [t.at(start) for t in txs]
        print(f"treatment {' + '.join(t.name for t in txs)} starts {how}", file=sys.stderr)
    res = simulate(
        patient,
        Combination(tuple(txs)) if txs else None,
        months=args.months,
        stochastic=args.stochastic,
        seed=args.seed,
        record_units=bool(args.units_plot),
    )
    _print_frame(res.summary().T.reset_index().rename(columns={"index": "metric", 0: "value"}))
    if args.csv:
        res.to_csv(args.csv)
        print(f"trajectory written to {args.csv}", file=sys.stderr)
    _save_fig(res.plot() if args.plot else None, args.plot)
    if args.units_plot:
        from .plotting import plot_motor_units

        _save_fig(plot_motor_units(res), args.units_plot)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .engine import simulate
    from .plotting import plot_trajectories

    _use_agg()
    patient = _patient(args)
    start, how = _resolve_start(patient, args, args.months)
    results = {"untreated": simulate(patient, months=args.months)}
    for name in args.treatments:
        results[name] = simulate(patient, get_treatment(name).at(start), months=args.months)
    rows = []
    for label, res in results.items():
        s = res.summary().iloc[0]
        rows.append(
            {
                "arm": label,
                "symptom_onset": s.symptom_onset_months,
                "event": s.event_months,
                "survival_from_onset": s.survival_from_symptom_onset,
                "months_to_half_function": s.months_to_half_function,
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    df["event_delay"] = df.event - df.event.iloc[0]
    print(f"treatments start {how}", file=sys.stderr)
    _print_frame(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
    if args.plot:
        onset = results["untreated"].symptom_onset[0]
        _save_fig(
            plot_trajectories(results, args.metric, marks={"treatment start": start - onset}),
            args.plot,
        )
    return 0


def cmd_cohort(args: argparse.Namespace) -> int:
    from .cohort import sample_cohort
    from .engine import simulate
    from .metrics import median_survival
    from .plotting import plot_cohort

    _use_agg()
    patients = sample_cohort(args.n, seed=args.seed)
    tx = _treatments(args.treatment)
    res = simulate(patients, tx or None, months=args.months, dt=0.5)
    summary = res.summary()
    surv = summary.survival_from_symptom_onset.to_numpy()
    observed = np.isfinite(surv)
    follow = np.where(observed, surv, args.months - summary.symptom_onset_months.to_numpy())
    ok = np.isfinite(follow)
    print(f"{args.n} virtual patients, horizon {args.months:g} months")
    print(
        f"  median survival from symptom onset (KM): {median_survival(follow[ok], observed[ok]):.1f} months"
    )
    print(
        f"  median first-year functional slope: {summary.alsfrs_slope_first_year.median():.2f} points/month"
    )
    print(
        f"  median motor-unit loss at symptom onset: {100 * summary.mu_loss_at_symptom_onset.median():.0f}%"
    )
    counts = summary.onset_region.value_counts().to_dict()
    print("  onset regions: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"per-patient summary written to {args.csv}", file=sys.stderr)
    _save_fig(plot_cohort(res) if args.plot else None, args.plot)
    return 0


def cmd_trial(args: argparse.Namespace) -> int:
    from .trial import TrialDesign, build_trial_pool, power_curve, run_trials

    _use_agg()
    design = TrialDesign(
        n_per_arm=args.n_per_arm,
        duration=args.duration,
        visit_interval=args.visit_interval,
        max_months_since_onset=args.max_months_since_onset,
        min_fvc=args.min_fvc,
        min_delta_fs=args.min_delta_fs,
        genotype=args.genotype,
        dropout_rate=args.dropout_rate,
    )
    sizes = args.sample_sizes or [args.n_per_arm]
    pool_size = args.pool_size or max(2000, 10 * max(sizes))
    print(f"screening virtual patients for a pool of {pool_size}...", file=sys.stderr)
    pool = build_trial_pool(
        Combination(tuple(_treatments(args.treatment))), design, pool_size=pool_size, seed=args.seed
    )
    print(f"{pool} (screen failure {100 * pool.screen_failure_rate:.0f}%)", file=sys.stderr)
    if args.sample_sizes:
        df = power_curve(pool, sizes, n_trials=args.trials, seed=args.seed + 1)
        _print_frame(df.pivot(index="n_per_arm", columns="endpoint", values="power").reset_index())
        if args.plot:
            from .plotting import plot_power_curve

            _save_fig(plot_power_curve(df), args.plot)
    else:
        res = run_trials(pool, args.trials, seed=args.seed + 1)
        df = res.power()
        _print_frame(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
    effects = pool.individual_effects()
    q = effects.alsfrs_preserved.quantile([0.1, 0.5, 0.9]).to_numpy()
    print(
        f"individual effect on functional score after {design.duration:g} months "
        f"(10th/50th/90th pct): {q[0]:+.1f} / {q[1]:+.1f} / {q[2]:+.1f} points",
        file=sys.stderr,
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import pandas as pd

    from .fitting import fit_patient
    from .plotting import plot_trajectories

    _use_agg()
    data = pd.read_csv(args.data)
    col = lambda name: data[name].to_numpy(dtype=float) if name else None  # noqa: E731
    fit = fit_patient(
        col(args.time_col),
        col(args.alsfrs_col),
        fvc=col(args.fvc_col),
        mune=col(args.mune_col),
        onset_region=args.onset_region,
        parameters=args.params,
    )
    print(f"fit {'converged' if fit.success else 'did NOT converge'} ({fit.message})")
    _print_frame(fit.confidence_intervals())
    print(f"RMSE (functional score): {fit.rmse_alsfrs:.2f} points over {fit.n_observations} visits")
    if args.save_patient:
        with open(args.save_patient, "w", encoding="utf-8") as fh:
            json.dump(fit.patient.to_dict(), fh, indent=2)
        print(f"fitted patient written to {args.save_patient}", file=sys.stderr)
    runs = {"fitted, untreated": fit.simulate(months_after_onset=args.horizon)}
    for name in args.forecast or []:
        runs[f"{name} from month {args.start_after_onset:g}"] = fit.counterfactual(
            get_treatment(name), args.start_after_onset, months_after_onset=args.horizon
        )
    for label, res in runs.items():
        ev = res.event_time[0] - fit.symptom_onset
        print(f"  {label:40s} respiratory failure at {ev:5.1f} months after symptom onset")
    if args.plot:
        fig = plot_trajectories(
            runs, "alsfrs", marks={"treatment start": args.start_after_onset} if args.forecast else None
        )
        fig.axes[0].scatter(
            fit.observations.months_since_onset,
            fit.observations.get("observed_alsfrs", np.nan),
            s=18,
            color="#0b0b0b",
            zorder=5,
            label="observed",
        )
        _save_fig(fig, args.plot)
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from .plotting import plot_sensitivity
    from .sensitivity import global_sensitivity

    _use_agg()
    res = global_sensitivity(base=_patient(args), n_samples=args.n, seed=args.seed)
    table = res.prcc.pivot(index="parameter", columns="output", values="prcc").reset_index()
    _print_frame(table, "{:+.2f}")
    if args.csv:
        res.prcc.to_csv(args.csv, index=False)
    _save_fig(plot_sensitivity(res, args.output) if args.plot else None, args.plot)
    return 0


def cmd_lag(args: argparse.Namespace) -> int:
    from .hidden_phase import response_lag

    df = response_lag(_patient(args), get_treatment(args.treatment))
    _print_frame(df, "{:.1f}")
    print(
        "\nnfl_lag / function_lag: months from treatment start until NfL is 15% lower / function 1 point "
        "better than untreated. A symptomatic drug changes function but never NfL.",
        file=sys.stderr,
    )
    return 0


def cmd_prevention(args: argparse.Namespace) -> int:
    from .hidden_phase import MonitoringRule, threshold_tradeoff, trigger_strategies

    _use_agg()
    rule = MonitoringRule(threshold=args.threshold, interval=args.interval, reference=args.reference)
    res = trigger_strategies(get_treatment(args.treatment), n=args.n, rule=rule, seed=args.seed)
    _print_frame(res.summary().reset_index(), "{:.2f}")
    tradeoff = threshold_tradeoff(rule=rule, n=args.n, seed=args.seed)
    print("\nNfL threshold trade-off:")
    _print_frame(tradeoff, "{:.3f}")
    if args.plot:
        from .plotting import plot_prevention

        _save_fig(plot_prevention(res.summary(), tradeoff), args.plot)
    return 0


# ----------------------------------------------------------------------
def _add_patient_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--phenotype", default="typical", choices=sorted(PHENOTYPES), help="preset patient")
    p.add_argument("--patient-json", help="JSON file with Patient fields (overrides --phenotype)")
    p.add_argument("--param", action="append", metavar="NAME=VALUE", help="override a patient parameter")


def _add_start_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", type=float, help="treatment start, months since disease onset")
    g.add_argument(
        "--start-after-onset",
        type=float,
        help="treatment start, months after symptom onset (default: at diagnosis)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mndsim",
        description="Mechanistic motor-unit degeneration simulator for ALS/MND research and teaching.",
    )
    parser.add_argument("--version", action="version", version=f"mndsim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list preset phenotypes and treatments")
    p.add_argument("what", nargs="?", default="all", choices=["all", "phenotypes", "treatments"])
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("simulate", help="simulate one patient")
    _add_patient_args(p)
    p.add_argument(
        "--treatment", action="append", choices=sorted(TREATMENTS), help="may repeat (combination)"
    )
    _add_start_args(p)
    p.add_argument("--months", type=float, default=72.0)
    p.add_argument("--stochastic", action="store_true", help="sample individual motor-unit fates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", help="write the full trajectory table")
    p.add_argument("--plot", help="write the overview figure (PNG/PDF/SVG)")
    p.add_argument("--units-plot", help="write the motor-unit raster figure")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("compare", help="counterfactual treatments for one patient")
    _add_patient_args(p)
    p.add_argument("treatments", nargs="+", choices=sorted(TREATMENTS))
    _add_start_args(p)
    p.add_argument("--months", type=float, default=96.0)
    p.add_argument("--metric", default="alsfrs", choices=["alsfrs", "fvc", "mvc", "mu_count"])
    p.add_argument("--csv")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("cohort", help="simulate a heterogeneous virtual cohort")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--months", type=float, default=150.0)
    p.add_argument("--treatment", action="append", choices=sorted(TREATMENTS))
    p.add_argument("--csv", help="write the per-patient summary")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_cohort)

    p = sub.add_parser("trial", help="in-silico randomised trial and power analysis")
    p.add_argument("--treatment", action="append", required=True, choices=sorted(TREATMENTS))
    p.add_argument("--n-per-arm", type=int, default=150)
    p.add_argument("--sample-sizes", type=int, nargs="+", help="compute a power curve over these sizes")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--visit-interval", type=float, default=1.0)
    p.add_argument("--max-months-since-onset", type=float, default=24.0)
    p.add_argument("--min-fvc", type=float, default=70.0)
    p.add_argument("--min-delta-fs", type=float, default=0.0, help="enrich for fast progressors")
    p.add_argument("--genotype", help="enrol only this genotype (e.g. SOD1)")
    p.add_argument("--dropout-rate", type=float, default=0.01)
    p.add_argument("--trials", type=int, default=500, help="replicate trials")
    p.add_argument("--pool-size", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv")
    p.add_argument("--plot", help="power-curve figure (with --sample-sizes)")
    p.set_defaults(func=cmd_trial)

    p = sub.add_parser("fit", help="fit a digital twin to a patient's measurements (CSV)")
    p.add_argument("data", help="CSV file with one row per visit")
    p.add_argument("--time-col", default="months_since_onset")
    p.add_argument("--alsfrs-col", default="alsfrs")
    p.add_argument("--fvc-col")
    p.add_argument("--mune-col")
    p.add_argument(
        "--onset-region", default="upper_limb", choices=["bulbar", "respiratory", "upper_limb", "lower_limb"]
    )
    p.add_argument("--params", nargs="+", default=["lmn_rate", "spread_rate"])
    p.add_argument("--forecast", nargs="+", choices=sorted(TREATMENTS), help="counterfactual treatments")
    p.add_argument("--start-after-onset", type=float, default=12.0)
    p.add_argument("--horizon", type=float, default=60.0, help="months after symptom onset to simulate")
    p.add_argument("--save-patient", help="write the fitted patient as JSON")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("lag", help="how fast NfL vs function respond to a treatment, by start time")
    _add_patient_args(p)
    p.add_argument("--treatment", default="neuroprotective", choices=sorted(TREATMENTS))
    p.set_defaults(func=cmd_lag)

    p = sub.add_parser("prevention", help="NfL-triggered treatment in presymptomatic gene carriers")
    p.add_argument("--treatment", default="gene_targeted", choices=sorted(TREATMENTS))
    p.add_argument("--n", type=int, default=300, help="virtual carriers")
    p.add_argument("--threshold", type=float, default=2.0, help="trigger at this multiple of baseline NfL")
    p.add_argument("--interval", type=float, default=3.0, help="months between blood tests")
    p.add_argument("--reference", default="personal", choices=["personal", "population"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot")
    p.set_defaults(func=cmd_prevention)

    p = sub.add_parser("sensitivity", help="global sensitivity analysis (LHS + PRCC)")
    _add_patient_args(p)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default="survival_from_symptom_onset",
        choices=[
            "symptom_onset_months",
            "mu_loss_at_symptom_onset",
            "alsfrs_slope_first_year",
            "survival_from_symptom_onset",
        ],
    )
    p.add_argument("--csv")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_sensitivity)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (KeyError, ValueError) as exc:
        parser.exit(2, f"mndsim: error: {exc}\n")
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
