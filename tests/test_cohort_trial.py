import numpy as np
import pytest
from scipy import stats

import mndsim
from mndsim import TrialDesign, build_trial_pool, run_trials
from mndsim.cohort import PopulationModel
from mndsim.treatment import TREATMENTS
from mndsim.trial import cafs_scores, logrank_test, welch_test


def test_cohort_sampling_is_reproducible_and_heterogeneous():
    a = mndsim.sample_cohort(300, seed=1)
    b = mndsim.sample_cohort(300, seed=1)
    assert a == b
    rates = np.array([p.lmn_rate for p in a])
    assert rates.std() / rates.mean() > 0.3
    onset = [p.onset_region for p in a]
    assert 0.15 < onset.count("bulbar") / len(onset) < 0.4
    assert len({p.seed for p in a}) == len(a)
    with pytest.raises(ValueError):
        PopulationModel(onset_region_probs={"toe": 1.0})


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 40), rng.normal(0.5, 2, 55)
    diff, p = welch_test(x, y)
    ref = stats.ttest_ind(x, y, equal_var=False)
    assert diff == pytest.approx(x.mean() - y.mean())
    assert p == pytest.approx(ref.pvalue)


@pytest.mark.skipif(not hasattr(stats, "logrank"), reason="scipy.stats.logrank needs SciPy >= 1.11")
def test_logrank_matches_scipy():
    rng = np.random.default_rng(1)
    t = rng.exponential(10, 120)
    group = np.arange(120) < 60
    t[group] *= 1.8
    event = t < 15
    t = np.minimum(t, 15)
    hr, p = logrank_test(t, event, group)
    ref = stats.logrank(
        stats.CensoredData.right_censored(t[group], ~event[group]),
        stats.CensoredData.right_censored(t[~group], ~event[~group]),
    )
    assert p == pytest.approx(ref.pvalue, rel=1e-6)
    assert hr < 1


def test_cafs_scores_are_antisymmetric():
    time = np.array([5.0, 12.0, 12.0, 3.0])
    died = np.array([True, False, False, True])
    func = np.array([-2.0, -0.5, -1.0, np.nan])
    s = cafs_scores(time, died, func)
    assert s.sum() == 0
    assert s[1] > s[2] > s[0] > s[3]


@pytest.fixture(scope="module")
def pool():
    design = TrialDesign(n_per_arm=60, duration=12)
    return build_trial_pool(TREATMENTS["neuroprotective"], design, pool_size=240, seed=3)


def test_pool_arms_identical_at_baseline(pool):
    assert pool.size == 240
    np.testing.assert_allclose(pool.items[0, :, 0], pool.items[1, :, 0])
    assert np.all(pool.months_since_onset <= pool.design.max_months_since_onset + 1e-9)
    assert np.all(pool.fvc[0, :, 0] >= pool.design.min_fvc)
    assert not np.isnan(pool.items).any()
    effects = pool.individual_effects()
    assert (effects.alsfrs_preserved > 0).mean() > 0.9
    assert 0 <= pool.screen_failure_rate < 1


def test_trial_detects_strong_effect(pool):
    res = run_trials(pool, 40, seed=1)
    power = res.power().set_index("endpoint").power
    assert power["alsfrs_slope"] > 0.5  # ~33% slowing, 60 per arm
    assert set(power.index) == set(mndsim.trial.ENDPOINTS)
    assert (res.table.alsfrs_slope_slowing > 0).mean() > 0.9
    with pytest.raises(ValueError):
        run_trials(pool, 2, n_per_arm=500)


def test_placebo_false_positive_rate_is_controlled():
    design = TrialDesign(n_per_arm=60, duration=9)
    pool = build_trial_pool(TREATMENTS["placebo"], design, pool_size=240, seed=4)
    np.testing.assert_array_equal(pool.items[0], pool.items[1])
    res = run_trials(pool, 200, seed=2)
    alpha_hat = (res.table.alsfrs_slope_p < 0.05).mean()
    assert alpha_hat < 0.12


def test_genotype_enrichment():
    design = TrialDesign(n_per_arm=20, duration=6, genotype="SOD1")
    pop = mndsim.DEFAULT_POPULATION.replace(genotype_probs={"sporadic": 0.7, "SOD1": 0.3})
    pool = build_trial_pool(TREATMENTS["gene_targeted"], design, population=pop, pool_size=60, seed=5)
    assert all(p.genotype == "SOD1" for p in pool.patients)


def test_power_curve(pool):
    df = mndsim.power_curve(pool, [20, 60], n_trials=20, seed=0)
    assert list(df.n_per_arm.unique()) == [20, 60]
    small = df[(df.n_per_arm == 20) & (df.endpoint == "alsfrs_slope")].power.iloc[0]
    large = df[(df.n_per_arm == 60) & (df.endpoint == "alsfrs_slope")].power.iloc[0]
    assert large >= small


def test_design_validation():
    with pytest.raises(ValueError):
        TrialDesign(n_per_arm=1)
    with pytest.raises(ValueError):
        TrialDesign(visit_interval=24, duration=12)
    with pytest.raises(ValueError):
        TrialDesign(mune_region="toe")
    assert TrialDesign(duration=12, visit_interval=3).visit_times.tolist() == [0, 3, 6, 9, 12]
