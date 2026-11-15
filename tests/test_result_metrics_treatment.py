import math

import numpy as np
import pytest

import mndsim
from mndsim import Treatment, simulate
from mndsim.metrics import auc, kaplan_meier, median_survival, slope, slopes, time_to_threshold
from mndsim.phenotypes import TYPICAL
from mndsim.scoring import fvc_percent, item_scores, observe_items
from mndsim.treatment import TREATMENTS, Combination, as_treatment_list, modifiers


# --- metrics ---------------------------------------------------------------
def test_time_to_threshold_interpolates():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.8, 0.4, 0.1])
    assert time_to_threshold(t, y, 0.6) == pytest.approx(1.5)
    assert math.isnan(time_to_threshold(t, y, 0.0))
    assert time_to_threshold(t, -y, -0.6, direction="above") == pytest.approx(1.5)
    with pytest.raises(ValueError):
        time_to_threshold(t, y, 0.5, direction="sideways")


def test_auc_and_slopes():
    t = np.linspace(0, 10, 11)
    assert auc(t, np.ones_like(t)) == pytest.approx(10.0)
    y = 3 - 0.5 * t
    assert slope(t, y) == pytest.approx(-0.5)
    Y = np.vstack([y, 2 * y, np.full_like(y, np.nan)])
    s = slopes(t, Y)
    assert s[0] == pytest.approx(-0.5) and s[1] == pytest.approx(-1.0) and np.isnan(s[2])
    Y[0, 5:] = np.nan
    assert slopes(t, Y)[0] == pytest.approx(-0.5)


def test_kaplan_meier():
    _, s = kaplan_meier(np.array([1, 2, 3, 4]), np.array([True, True, False, True]))
    assert s[0] == 1.0 and s[-1] == 0.0
    assert median_survival(np.array([1, 2, 3, 4]), np.array([True] * 4)) == 2.0
    assert math.isnan(median_survival(np.array([1, 2]), np.array([False, False])))


# --- scoring ---------------------------------------------------------------
def test_item_scores_bounds_and_monotonicity():
    caps = np.linspace(0, 1, 50)
    full = np.stack([caps] * 4, axis=-1)
    items = item_scores(full)
    assert items.shape == (50, 12)
    assert np.allclose(items[0], 0.0) and np.allclose(items[-1], 4.0)
    assert np.all(np.diff(items, axis=0) >= -1e-12)
    rng = np.random.default_rng(0)
    obs = observe_items(items, rng)
    assert set(np.unique(obs)) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert fvc_percent(np.array([0.5]))[0] == 50.0


# --- treatment -------------------------------------------------------------
def test_efficacy_profile():
    tx = Treatment(lmn_protection=0.5, start=10, stop=20, ramp=2, washout=3)
    e = tx.efficacy(np.array([0.0, 10.0, 12.0, 20.0, 23.0, 100.0]))
    assert e[0] == 0 and e[1] == 0
    assert e[2] == pytest.approx(1 - math.exp(-1))
    assert e[4] == pytest.approx(e[3] * math.exp(-1))
    assert e[5] == pytest.approx(0, abs=1e-9)
    step = Treatment(lmn_protection=0.5, ramp=0, start=1).efficacy(np.array([0.99, 1.0]))
    assert list(step) == [0.0, 1.0]


def test_combination_composes_multiplicatively():
    a = Treatment(name="a", lmn_protection=0.5, ramp=0)
    b = Treatment(name="b", lmn_protection=0.5, ramp=0)
    combo = a + b
    assert isinstance(combo, Combination) and combo.name == "a + b"
    mods = modifiers(as_treatment_list(combo), np.array([1.0]), np.array(["sporadic"], dtype=object))
    assert mods.lmn[0] == pytest.approx(0.25)
    assert len(as_treatment_list([a, combo])) == 3
    assert TREATMENTS["placebo"].is_null and not TREATMENTS["neuroprotective"].is_null


def test_get_unknown_names():
    with pytest.raises(KeyError):
        mndsim.get_treatment("magic")
    with pytest.raises(KeyError):
        mndsim.get_phenotype("zombie")


# --- result ----------------------------------------------------------------
@pytest.fixture(scope="module")
def batch():
    return simulate(list(mndsim.PHENOTYPES.values()), months=90)


def test_summary_and_frames(batch):
    s = batch.summary()
    assert len(s) == len(mndsim.PHENOTYPES)
    for col in (
        "symptom_onset_months",
        "survival_from_symptom_onset",
        "delta_fs",
        "mu_loss_at_symptom_onset",
    ):
        assert col in s
    healthy = s[s.patient == "healthy"].iloc[0]
    assert np.isnan(healthy.symptom_onset_months)
    df = batch.to_frame(patient=0)
    assert len(df) == len(batch.time)
    assert {"alsfrs", "fvc", "mvc_bulbar", "item_speech", "enlargement_upper_limb"} <= set(df.columns)
    assert len(batch.to_frame()) == len(batch.time) * len(batch)
    assert "SimulationResult" in repr(batch)


def test_subset_and_value_at(batch):
    sub = batch.subset([0, 2])
    assert len(sub) == 2 and sub.patients[1] is batch.patients[2]
    np.testing.assert_array_equal(sub.alsfrs[1], batch.alsfrs[2])
    v = batch.value_at("mvc", np.zeros(len(batch)), region="bulbar")
    assert np.allclose(v, 1.0)
    with pytest.raises(ValueError):
        batch.value_at("mvc", 0.0)


def test_mup_size_and_mu_loss(batch):
    assert np.allclose(batch.mup_size[:, 0], 1.0)
    assert np.all((batch.mu_loss >= -1e-9) & (batch.mu_loss <= 1 + 1e-9))


def test_to_csv(tmp_path):
    res = simulate(TYPICAL, months=6)
    path = tmp_path / "out.csv"
    res.to_csv(path)
    assert path.read_text().startswith("patient,")
