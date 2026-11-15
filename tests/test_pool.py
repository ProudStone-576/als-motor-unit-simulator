import numpy as np
import pytest

from mndsim import DEFAULT_CONFIG, MotorUnitPool, PhysiologyConfig
from mndsim.pool import mean_force_factor


@pytest.fixture(scope="module")
def pool():
    return MotorUnitPool.from_config(DEFAULT_CONFIG)


def test_size_principle(pool):
    assert pool.n_units == DEFAULT_CONFIG.n_units
    assert np.isclose(pool.fibers.sum(), 1.0)
    assert np.all(np.diff(pool.fibers) > 0), "units ordered small to large"
    assert np.all(np.diff(pool.threshold) > 0), "larger units recruited later"
    assert np.all(np.diff(pool.contraction_time) < 0), "larger units are faster"
    assert pool.fibers[-1] / pool.fibers[0] == pytest.approx(DEFAULT_CONFIG.twitch_force_range, rel=0.05)
    assert pool.contraction_time[0] / pool.contraction_time[-1] == pytest.approx(3.0, rel=0.05)


def test_unit_types_follow_size(pool):
    assert list(np.unique(pool.unit_type)) == [0, 1, 2]
    assert np.all(np.diff(pool.unit_type) >= 0)
    counts = np.bincount(pool.unit_type) / pool.n_units
    assert np.allclose(counts, DEFAULT_CONFIG.type_fractions, atol=0.02)


def test_force_is_normalised_and_monotone(pool):
    _, force = pool.force_excitation_curve(201)
    assert force[0] == 0.0
    assert force[-1] == pytest.approx(1.0)
    assert np.all(np.diff(force) >= -1e-12)


def test_recruitment(pool):
    rates = pool.firing_rates(pool.threshold[10] + 1e-9)
    assert np.all(rates[:11] > 0) and np.all(rates[11:] == 0)
    peak = pool.firing_rates(pool.max_excitation)
    assert np.allclose(peak, pool.peak_rate)


def test_fast_force_matches_exact(pool):
    e = np.linspace(0, pool.max_excitation, 301)
    fibers = np.broadcast_to(pool.fibers, (e.size, pool.n_units))
    exact = pool.force(fibers, e)
    fast = pool.fast_force(fibers, e)
    assert np.max(np.abs(exact - fast)) < 5e-3


def test_mean_force_factor_continuous_and_saturating():
    left = mean_force_factor(0.4 - 1e-9)
    right = mean_force_factor(0.4 + 1e-9)
    assert left == pytest.approx(right, rel=1e-6)
    x = np.linspace(0, 10, 500)
    y = mean_force_factor(x)
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == pytest.approx(mean_force_factor(50.0), rel=1e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        PhysiologyConfig(type_fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        PhysiologyConfig(n_units=2)
    with pytest.raises(ValueError):
        PhysiologyConfig(type_sprouting=(1.0, 1.5, 0.2))
