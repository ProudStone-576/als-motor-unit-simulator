"""Regenerate every figure used in the README and docs.

    python examples/make_figures.py            # writes docs/images/*.png

Takes about a minute (the power analysis dominates).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

import mndsim
from mndsim import plotting
from mndsim.phenotypes import TYPICAL

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Overview of a typical disease course.
    res = mndsim.simulate(TYPICAL, months=72)
    plotting.save(res.plot(), OUT / "overview.png")

    # 2. Individual motor-unit fates (stochastic) - FF units are lost first.
    stoch = mndsim.simulate(TYPICAL, months=60, stochastic=True, seed=7, record_units=True)
    plotting.save(plotting.plot_motor_units(stoch), OUT / "motor_units.png")

    # 3. The clinical threshold: compensation hides motor-unit loss.
    comp = {
        ("no reinnervation" if e == 1 else f"units may grow {e:g}x"): mndsim.simulate(
            TYPICAL.replace(max_enlargement=e), months=96
        )
        for e in (1.0, 2.5, 5.0, 8.0)
    }
    plotting.save(plotting.plot_compensation(comp), OUT / "compensation.png")

    # 4. Counterfactual treatment timing.
    onset = res.symptom_onset[0]
    tx = mndsim.TREATMENTS["neuroprotective"]
    runs = {"untreated": res}
    for label, start in (("at symptom onset", 0.0), ("at diagnosis (+12 mo)", 12.0), ("+24 mo", 24.0)):
        runs[f"neuroprotection {label}"] = mndsim.simulate(TYPICAL, tx.at(onset + start), months=72)
    plotting.save(
        plotting.plot_trajectories(runs, "alsfrs", title="Same patient, same drug, different start"),
        OUT / "treatment_timing.png",
    )

    # 5. Virtual cohort.
    cohort = mndsim.simulate(mndsim.sample_cohort(400, seed=3), months=150, dt=0.5)
    plotting.save(plotting.plot_cohort(cohort), OUT / "cohort.png")

    # 6. Global sensitivity analysis.
    sa = mndsim.global_sensitivity(n_samples=400, seed=1)
    plotting.save(
        plotting.plot_sensitivity(
            sa, "survival_from_symptom_onset", title="What drives survival after symptom onset?"
        ),
        OUT / "sensitivity.png",
    )

    # 7. Power analysis for an in-silico trial.
    design = mndsim.TrialDesign(n_per_arm=200, duration=12)
    pool = mndsim.build_trial_pool(mndsim.TREATMENTS["anti_excitotoxic"], design, pool_size=2500, seed=11)
    curve = mndsim.power_curve(pool, [25, 50, 100, 150, 200, 300], n_trials=300, seed=2)
    plotting.save(
        plotting.plot_power_curve(curve, title="Anti-excitotoxic archetype, 12-month trial"),
        OUT / "power.png",
    )
    print(curve.pivot(index="n_per_arm", columns="endpoint", values="power").round(2))

    # 8. The hidden phase: loss, compensation and NfL around symptom onset.
    plotting.save(plotting.plot_hidden_phase(res), OUT / "hidden_phase.png")

    # 9. Endpoint fingerprints: NfL vs function by trial length (60 per arm).
    panels = {}
    for label, name in (
        ("Neuroprotection (disease-modifying)", "neuroprotective"),
        ("Symptomatic only", "symptomatic"),
    ):
        pool18 = mndsim.build_trial_pool(
            mndsim.TREATMENTS[name], mndsim.TrialDesign(n_per_arm=60, duration=18), pool_size=1500, seed=5
        )
        panels[label] = mndsim.power_by_duration(
            pool18, [3, 6, 9, 12, 18], n_per_arm=60, n_trials=300, seed=6
        )
    plotting.save(plotting.plot_power_by_duration(panels, ("nfl", "alsfrs_slope")), OUT / "fingerprints.png")
    for label, df in panels.items():
        print(label)
        print(
            df[df.endpoint.isin(["nfl", "alsfrs_slope"])]
            .pivot(index="duration", columns="endpoint", values="power")
            .round(2)
        )

    # 10. NfL-triggered treatment in presymptomatic SOD1 carriers.
    from mndsim import hidden_phase as hp

    strategies = hp.trigger_strategies(mndsim.TREATMENTS["gene_targeted"], n=400, seed=1)
    tradeoff = hp.threshold_tradeoff((1.3, 1.5, 1.75, 2.0, 2.5, 3.0), n=400, seed=1)
    plotting.save(plotting.plot_prevention(strategies.summary(), tradeoff), OUT / "prevention.png")
    print(strategies.summary().round(2))
    print(tradeoff.round(3))
    print(f"wrote figures to {OUT}", file=sys.stderr)
    _ = np  # keep the import for interactive tinkering


if __name__ == "__main__":
    main()
