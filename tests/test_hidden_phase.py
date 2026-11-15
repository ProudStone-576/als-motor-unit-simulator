import numpy as np
import pytest

import mndsim
from mndsim import hidden_phase as hp
from mndsim.engine import healthy_nfl
from mndsim.phenotypes import FAST_PROGRESSOR, HEALTHY, SLOW_PROGRESSOR, TYPICAL
from mndsim.treatment import TREATMENTS


# --- NfL in the engine ------------------------------------------------------
def test_healthy_nfl_is_flat_and_age_dependent():
    young = mndsim.simulate(HEALTHY.replace(age_at_onset=40), months=24)
    old = mndsim.simulate(HEALTHY.replace(age_at_onset=75), months=24)
    assert np.allclose(young.nfl, healthy_nfl(40))
    assert np.allclose(old.nfl, healthy_nfl(75))
    assert healthy_nfl(75) > healthy_nfl(40)


def test_nfl_rises_before_symptoms_and_tracks_speed():
    res = mndsim.simulate([TYPICAL, FAST_PROGRESSOR, SLOW_PROGRESSOR], months=150)
    s = res.summary()
    assert s.nfl_lead_time[0] > 0, "NfL doubles before symptoms"
    assert 2.0 < s.nfl_fold_at_diagnosis[0] < 15.0
    assert s.nfl_at_diagnosis[1] > s.nfl_at_diagnosis[0] > s.nfl_at_diagnosis[2]


def test_population_nfl_is_prognostic():
    s = mndsim.simulate(mndsim.sample_cohort(150, seed=3), months=150, dt=0.5).summary()
    ok = np.isfinite(s.nfl_at_diagnosis) & np.isfinite(s.delta_fs) & (s.delta_fs > 0)
    rho = np.corrcoef(np.log(s.nfl_at_diagnosis[ok]), np.log(s.delta_fs[ok]))[0, 1]
    assert rho > 0.3
    assert np.nanmedian(s.nfl_lead_time) > 0


def test_symptomatic_treatment_leaves_nfl_untouched():
    base = mndsim.simulate(TYPICAL, months=60)
    sym = mndsim.simulate(TYPICAL, TREATMENTS["symptomatic"].at(10), months=60)
    np.testing.assert_allclose(sym.nfl, base.nfl)
    prot = mndsim.simulate(TYPICAL, TREATMENTS["neuroprotective"].at(10), months=60)
    i = np.searchsorted(base.time, 16)
    assert prot.nfl[0, i] < 0.9 * base.nfl[0, i]


def test_stochastic_nfl_matches_mean_field_on_average():
    p = TYPICAL.replace(seed=4)
    mf = mndsim.simulate(p, months=30)
    st = mndsim.simulate([p] * 60, months=30, stochastic=True, seed=2)
    assert np.allclose(st.nfl.mean(0), mf.nfl[0], rtol=0.15)


# --- response lag -----------------------------------------------------------
def test_response_lag_fingerprints():
    prot = hp.response_lag(TYPICAL, TREATMENTS["neuroprotective"])
    sym = hp.response_lag(TYPICAL, TREATMENTS["symptomatic"])
    assert prot.start.iloc[0].startswith("presymptomatic")
    assert np.isfinite(prot.nfl_lag).any() and (prot.max_nfl_reduction > 0.1).all()
    assert sym.nfl_lag.isna().all() and (sym.max_nfl_reduction < 1e-9).all()
    assert prot.event_gain.iloc[0] > prot.event_gain.iloc[-1], "earlier is better"
    traj = hp.lag_trajectories(TYPICAL, TREATMENTS["neuroprotective"], 20.0)
    assert traj.nfl_ratio.iloc[0] == pytest.approx(1.0)
    assert traj.nfl_ratio.min() < 0.85


# --- NfL-triggered treatment -------------------------------------------------
def test_trigger_strategies():
    res = hp.trigger_strategies(TREATMENTS["gene_targeted"], n=60, seed=2)
    summary = res.summary()
    assert set(summary.index) == {"untreated", "nfl_trigger", "symptom_onset", "diagnosis"}
    assert summary.loc["nfl_trigger", "treated_fraction"] > 0.9
    assert (
        summary.loc["nfl_trigger", "median_start_vs_symptoms"]
        < summary.loc["diagnosis", "median_start_vs_symptoms"]
    )
    assert (
        summary.loc["symptom_onset", "median_event_gain"] > summary.loc["diagnosis", "median_event_gain"] > 0
    )
    assert (
        summary.loc["nfl_trigger", "median_function_at_+24mo"]
        > summary.loc["untreated", "median_function_at_+24mo"]
    )


def test_false_triggers():
    personal = [hp.false_trigger_rate(hp.MonitoringRule(threshold=t), n=600) for t in (1.3, 1.75, 2.5)]
    assert personal[0] > personal[1] > personal[2]
    pop = hp.false_trigger_rate(hp.MonitoringRule(threshold=2.0, reference="population"), n=600)
    own = hp.false_trigger_rate(hp.MonitoringRule(threshold=2.0, reference="personal"), n=600)
    assert pop > own
    with pytest.raises(ValueError):
        hp.MonitoringRule(reference="neighbour")
    with pytest.raises(ValueError):
        hp.MonitoringRule(threshold=0.9)


def test_threshold_tradeoff_monotone():
    df = hp.threshold_tradeoff((1.5, 2.0, 3.0), n=80, seed=1)
    assert np.all(np.diff(df.median_lead_time) <= 1e-9)
    assert np.all(np.diff(df.false_trigger_rate_10y) <= 1e-9)


# --- trials with NfL ----------------------------------------------------------
@pytest.fixture(scope="module")
def pool():
    design = mndsim.TrialDesign(n_per_arm=50, duration=12)
    return mndsim.build_trial_pool(TREATMENTS["neuroprotective"], design, pool_size=220, seed=7)


def test_nfl_endpoint_and_duration(pool):
    assert pool.nfl.shape == pool.fvc.shape and np.all(pool.nfl > 0)
    df = mndsim.power_by_duration(pool, [3, 6, 12], n_trials=30, seed=1)
    nfl = df[df.endpoint == "nfl"].set_index("duration")
    assert (nfl.median_effect < 1.0).all(), "treated/control NfL ratio below 1"
    assert nfl.power.max() > 0.3
    with pytest.raises(ValueError):
        mndsim.run_trials(pool, 2, duration=24)


def test_nfl_enrichment():
    design = mndsim.TrialDesign(n_per_arm=20, duration=6, min_nfl_fold=5.0)
    pool = mndsim.build_trial_pool(TREATMENTS["placebo"], design, pool_size=60, seed=2)
    healthy = healthy_nfl(np.array([p.age_at_onset for p in pool.patients]))
    assert np.all(pool.nfl[0, :, 0] / healthy >= 5.0 - 1e-6)


def test_cli_lag_and_prevention(tmp_path, capsys):
    from mndsim.cli import main

    assert main(["lag", "--phenotype", "typical", "--treatment", "symptomatic"]) == 0
    assert "function_lag" in capsys.readouterr().out
    assert main(["prevention", "--n", "40", "--plot", str(tmp_path / "p.png")]) == 0
    assert "nfl_trigger" in capsys.readouterr().out
    assert (tmp_path / "p.png").exists()


def test_new_plots_render():
    from mndsim import plotting

    res = mndsim.simulate(TYPICAL, months=80)
    assert plotting.plot_hidden_phase(res) is not None
    pool = mndsim.build_trial_pool(
        TREATMENTS["neuroprotective"], mndsim.TrialDesign(n_per_arm=20, duration=6), pool_size=60, seed=1
    )
    df = mndsim.power_by_duration(pool, [3, 6], n_trials=5)
    assert plotting.plot_power_by_duration({"a": df}) is not None
    strat = hp.trigger_strategies(TREATMENTS["gene_targeted"], n=20, seed=1)
    trade = hp.threshold_tradeoff((1.5, 2.0), n=20)
    assert plotting.plot_prevention(strat.summary(), trade) is not None
    with pytest.raises(ValueError):
        plotting.plot_hidden_phase(mndsim.simulate(HEALTHY, months=12))
