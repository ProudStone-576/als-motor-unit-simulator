import json

import numpy as np
import pandas as pd
import pytest

import mndsim
from mndsim import plotting
from mndsim.cli import main
from mndsim.phenotypes import TYPICAL
from mndsim.sensitivity import prcc


# --- fitting ---------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_patient():
    truth = TYPICAL.replace(lmn_rate=0.12, spread_rate=0.06, onset_region="bulbar", seed=0)
    res = mndsim.simulate(truth, months=120, dt=0.5, record_every=0.5)
    t = np.arange(3.0, 27.0, 3.0)
    on = res.symptom_onset[0]
    alsfrs = np.interp(on + t, res.time, res.alsfrs[0])
    fvc = np.interp(on + t, res.time, res.fvc[0])
    return truth, t, alsfrs, fvc


def test_fit_recovers_parameters(synthetic_patient):
    truth, t, alsfrs, fvc = synthetic_patient
    fit = mndsim.fit_patient(t, alsfrs, fvc=fvc, onset_region="bulbar")
    assert fit.success
    assert fit.parameters["lmn_rate"] == pytest.approx(truth.lmn_rate, rel=0.1)
    assert fit.parameters["spread_rate"] == pytest.approx(truth.spread_rate, rel=0.1)
    assert fit.rmse_alsfrs < 0.5
    ci = fit.confidence_intervals()
    assert np.all(ci.lower <= ci.estimate) and np.all(ci.estimate <= ci.upper)
    treated = fit.counterfactual(mndsim.TREATMENTS["neuroprotective"], start_after_onset=3.0)
    untreated = fit.simulate()
    assert treated.event_time[0] > untreated.event_time[0]


def test_fit_validation():
    with pytest.raises(ValueError):
        mndsim.fit_patient([1, 2, 3])
    with pytest.raises(ValueError):
        mndsim.fit_patient([1, 2], [40, 39, 38])
    with pytest.raises(ValueError):
        mndsim.fit_patient([1, 2, 3], [40, 39, 38], parameters=["genotype"])


# --- sensitivity -----------------------------------------------------------
def test_prcc_identifies_the_driver():
    rng = np.random.default_rng(0)
    X = rng.random((400, 3))
    y = np.exp(3 * X[:, 1]) + 0.05 * rng.random(400)
    coef, p = prcc(X, y)
    assert coef[1] > 0.95 and abs(coef[0]) < 0.2 and abs(coef[2]) < 0.2
    assert p[1] < 1e-6


def test_global_sensitivity_and_sweep():
    sa = mndsim.global_sensitivity(n_samples=64, seed=0, months=150)
    assert len(sa.samples) == 64
    rank = sa.ranking("mu_loss_at_symptom_onset")
    assert rank.parameter.iloc[0] in ("max_enlargement", "lmn_rate")
    df = mndsim.sweep("lmn_rate", [0.04, 0.08, 0.16])
    assert np.all(np.diff(df.symptom_onset_months) < 0)
    with pytest.raises(ValueError):
        mndsim.global_sensitivity({"genotype": (1, 2)})


# --- plotting --------------------------------------------------------------
def test_all_plots_render():
    res = mndsim.simulate(TYPICAL, months=60, record_units=True)
    figs = [
        res.plot(),
        res.plot(aligned=True),
        plotting.plot_motor_units(res),
        plotting.plot_trajectories({"a": res, "b": res}, "fvc", marks={"x": 3.0}),
        plotting.plot_trajectories({"a": res}, "mvc"),
        plotting.plot_compensation({"a": res}),
        plotting.plot_cohort(mndsim.simulate(mndsim.sample_cohort(30, seed=1), months=60)),
        plotting.plot_kaplan_meier({"a": (np.array([1.0, 2, 3]), np.array([True, False, True]))}),
        plotting.plot_power_curve(
            pd.DataFrame({"n_per_arm": [10, 20], "endpoint": ["alsfrs_slope"] * 2, "power": [0.2, 0.4]})
        ),
        plotting.plot_sensitivity(
            mndsim.global_sensitivity(n_samples=24, months=100), "symptom_onset_months"
        ),
    ]
    assert all(hasattr(f, "savefig") for f in figs)
    with pytest.raises(ValueError):
        plotting.plot_motor_units(mndsim.simulate(TYPICAL, months=6))


# --- CLI -------------------------------------------------------------------
def test_cli_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "typical" in out and "neuroprotective" in out


def test_cli_simulate_and_compare(tmp_path, capsys):
    csv, png = tmp_path / "t.csv", tmp_path / "t.png"
    assert (
        main(
            [
                "simulate",
                "--phenotype",
                "fast_progressor",
                "--treatment",
                "neuroprotective",
                "--months",
                "48",
                "--csv",
                str(csv),
                "--plot",
                str(png),
            ]
        )
        == 0
    )
    assert csv.exists() and png.exists()
    assert (
        main(
            [
                "compare",
                "neuroprotective",
                "symptomatic",
                "--start-after-onset",
                "3",
                "--months",
                "60",
                "--plot",
                str(tmp_path / "c.png"),
            ]
        )
        == 0
    )
    assert "event_delay" in capsys.readouterr().out
    patient = tmp_path / "p.json"
    patient.write_text(json.dumps(TYPICAL.replace(lmn_rate=0.2).to_dict()))
    assert (
        main(["simulate", "--patient-json", str(patient), "--param", "spread_rate=0.1", "--months", "24"])
        == 0
    )


def test_cli_cohort_trial_sensitivity(tmp_path, capsys):
    assert main(["cohort", "--n", "40", "--months", "100", "--csv", str(tmp_path / "c.csv")]) == 0
    assert "median survival" in capsys.readouterr().out
    assert (
        main(
            [
                "trial",
                "--treatment",
                "neuroprotective",
                "--n-per-arm",
                "30",
                "--trials",
                "10",
                "--pool-size",
                "120",
                "--duration",
                "6",
            ]
        )
        == 0
    )
    assert "alsfrs_slope" in capsys.readouterr().out
    assert main(["sensitivity", "--n", "24", "--plot", str(tmp_path / "s.png")]) == 0


def test_cli_fit(tmp_path, synthetic_patient, capsys):
    _, t, alsfrs, fvc = synthetic_patient
    data = tmp_path / "visits.csv"
    pd.DataFrame({"months_since_onset": t, "alsfrs": alsfrs, "fvc": fvc}).to_csv(data, index=False)
    assert (
        main(
            [
                "fit",
                str(data),
                "--fvc-col",
                "fvc",
                "--onset-region",
                "bulbar",
                "--forecast",
                "neuroprotective",
                "--start-after-onset",
                "3",
                "--save-patient",
                str(tmp_path / "twin.json"),
                "--plot",
                str(tmp_path / "f.png"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "converged" in out and "lmn_rate" in out
    twin = mndsim.Patient.from_dict(json.loads((tmp_path / "twin.json").read_text()))
    assert twin.onset_region == "bulbar"


def test_cli_errors():
    with pytest.raises(SystemExit):
        main(["simulate", "--param", "nonsense"])
    with pytest.raises(SystemExit):
        main(["simulate", "--param", "lmn_rate=-1"])
