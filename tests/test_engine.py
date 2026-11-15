import math

import numpy as np
import pytest

import mndsim
from mndsim import DEFAULT_CONFIG, Patient, Treatment, simulate
from mndsim.phenotypes import HEALTHY, PHENOTYPES, SOD1_FAMILIAL, TYPICAL
from mndsim.treatment import TREATMENTS


@pytest.fixture(scope="module")
def typical():
    return simulate(TYPICAL, months=96)


def test_healthy_patient_stays_healthy():
    # Regression: the original simulator divided by the degeneration rate,
    # so a rate of zero crashed.
    res = simulate(HEALTHY, months=36)
    assert np.allclose(res.mvc, 1.0)
    assert np.allclose(res.mu_count, DEFAULT_CONFIG.n_units)
    assert np.allclose(res.alsfrs, 48.0)
    assert np.isnan(res.symptom_onset[0]) and np.isnan(res.event_time[0])


def test_zero_rates_with_treatment_do_not_crash():
    p = TYPICAL.replace(lmn_rate=0.0, umn_rate=0.0)
    res = simulate(p, TREATMENTS["neuroprotective"], months=12)
    assert np.allclose(res.mvc, 1.0)


def test_fibre_conservation():
    # With perfectly transmitting denervating units, every muscle fibre is
    # either innervated (cmap), orphaned (denervated) or lost.
    cfg = DEFAULT_CONFIG.replace(denervating_transmission=1.0)
    for stochastic in (False, True):
        res = simulate(PHENOTYPES["fast_progressor"], months=60, config=cfg, stochastic=stochastic, seed=1)
        total = res.cmap + res.denervated + res.lost_fibers
        assert np.allclose(total, 1.0, atol=1e-9)


def test_untreated_disease_is_progressive(typical):
    assert np.all(np.diff(typical.mu_count, axis=1) <= 1e-9)
    assert np.all(np.diff(typical.umn, axis=1) <= 1e-12)
    assert typical.alsfrs[0, -1] < typical.alsfrs[0, 0]
    assert 0 < typical.symptom_onset[0] < typical.event_time[0]


def test_typical_patient_matches_calibration_targets(typical):
    s = typical.summary().iloc[0]
    assert 20 <= s.survival_from_symptom_onset <= 42  # roughly 2-3.5 years
    assert -1.2 <= s.alsfrs_slope_first_year <= -0.5
    assert 0.35 <= s.mu_loss_at_symptom_onset <= 0.7  # loss hidden by reinnervation
    assert 34 <= s.alsfrs_at_diagnosis <= 42
    assert s.symptom_region == "upper_limb"


def test_selective_vulnerability_ff_first(typical):
    r = mndsim.REGIONS.index("upper_limb")
    surv = typical.type_survival[0, :, r]  # (T, 3) for S, FR, FF
    mid = np.searchsorted(typical.time, typical.symptom_onset[0])
    assert surv[mid, 2] < surv[mid, 1] < surv[mid, 0]


def test_reinnervation_hides_motor_unit_loss():
    rich = simulate(TYPICAL.replace(max_enlargement=6.0), months=96)
    poor = simulate(TYPICAL.replace(max_enlargement=1.0), months=96)
    r = mndsim.REGIONS.index("upper_limb")
    for res in (rich, poor):
        assert np.nanmax(res.enlargement[0, :, r]) >= 1.0
    assert rich.summary().mu_loss_at_symptom_onset[0] > poor.summary().mu_loss_at_symptom_onset[0] + 0.2
    assert np.nanmax(rich.enlargement[0, :, r]) > 3.0
    assert np.nanmax(poor.enlargement[0, :, r]) < 1.05


def test_motor_unit_number_falls_before_strength(typical):
    r = mndsim.REGIONS.index("upper_limb")
    t = typical.time
    t_mune = mndsim.metrics.time_to_threshold(t, typical.mu_count[0, :, r] / 120, 0.8)
    t_cmap = mndsim.metrics.time_to_threshold(t, typical.cmap[0, :, r], 0.8)
    t_force = mndsim.metrics.time_to_threshold(t, typical.mvc[0, :, r], 0.8)
    assert t_mune < t_cmap < t_force


def test_placebo_equals_untreated(typical):
    placebo = simulate(TYPICAL, TREATMENTS["placebo"], months=96)
    np.testing.assert_array_equal(placebo.alsfrs, typical.alsfrs)
    np.testing.assert_array_equal(placebo.event_time, typical.event_time)


@pytest.mark.parametrize(
    "name", ["neuroprotective", "anti_excitotoxic", "nmj_stabilizer", "sprouting_enhancer"]
)
def test_disease_modifying_treatments_delay_event(typical, name):
    treated = simulate(TYPICAL, TREATMENTS[name].at(typical.symptom_onset[0]), months=96)
    assert treated.event_time[0] > typical.event_time[0] + 1.0


def test_earlier_treatment_is_better(typical):
    tx = TREATMENTS["neuroprotective"]
    events = [simulate(TYPICAL, tx.at(start), months=120).event_time[0] for start in (0.0, 15.0, 30.0)]
    assert events[0] > events[1] > events[2] > typical.event_time[0]


def test_symptomatic_treatment_does_not_modify_degeneration(typical):
    sym = simulate(TYPICAL, TREATMENTS["symptomatic"], months=96)
    np.testing.assert_allclose(sym.mu_count, typical.mu_count)
    assert np.all(sym.mvc >= typical.mvc - 1e-12)
    assert sym.mvc[0, -1, 0] > typical.mvc[0, -1, 0]


def test_gene_targeted_only_affects_matching_genotype(typical):
    tx = TREATMENTS["gene_targeted"].at(0.0)
    sporadic = simulate(TYPICAL, tx, months=96)
    np.testing.assert_array_equal(sporadic.alsfrs, typical.alsfrs)
    base = simulate(SOD1_FAMILIAL, months=120)
    treated = simulate(SOD1_FAMILIAL, tx, months=120)
    assert treated.event_time[0] > base.event_time[0] + 12


def test_time_origin_shifts_treatment():
    tx = TREATMENTS["neuroprotective"].at(0.0)
    a = simulate(TYPICAL, tx.at(20.0), months=80)
    b = simulate(TYPICAL, tx, months=80, time_origin=20.0)
    np.testing.assert_allclose(a.alsfrs, b.alsfrs)


def test_stochastic_is_reproducible_and_integer():
    a = simulate(TYPICAL, months=48, stochastic=True, seed=3)
    b = simulate(TYPICAL, months=48, stochastic=True, seed=3)
    c = simulate(TYPICAL, months=48, stochastic=True, seed=4)
    np.testing.assert_array_equal(a.mu_count, b.mu_count)
    assert not np.array_equal(a.mu_count, c.mu_count)
    assert np.allclose(a.mu_count, np.round(a.mu_count))


def test_stochastic_average_matches_mean_field():
    p = TYPICAL.replace(seed=11)
    mean_field = simulate(p, months=36)
    runs = simulate([p] * 60, months=36, stochastic=True, seed=5)
    avg = runs.mu_count.mean(axis=0)
    r = mndsim.REGIONS.index("upper_limb")
    assert np.allclose(avg[:, r], mean_field.mu_count[0, :, r], atol=3.0)


def test_batch_and_chunking_are_consistent():
    patients = mndsim.sample_cohort(37, seed=2)
    whole = simulate(patients, months=40, chunk_size=1000)
    chunked = simulate(patients, months=40, chunk_size=7)
    single = simulate(patients[5], months=40)
    np.testing.assert_array_equal(whole.alsfrs, chunked.alsfrs)
    np.testing.assert_allclose(whole.alsfrs[5], single.alsfrs[0])


def test_time_step_convergence():
    fine = simulate(TYPICAL, months=80, dt=0.125)
    coarse = simulate(TYPICAL, months=80, dt=0.5)
    assert abs(fine.event_time[0] - coarse.event_time[0]) < 1.0
    assert np.max(np.abs(fine.alsfrs - coarse.alsfrs)) < 2.0


def test_early_stopping_keeps_needed_values():
    patients = mndsim.sample_cohort(40, seed=8)
    full = simulate(patients, months=120, dt=0.5)
    early = simulate(patients, months=120, dt=0.5, stop_after_onset=6.0, chunk_size=8)
    needed = full.time[None, :] <= (full.symptom_onset + 6.0)[:, None]
    assert np.all(np.isfinite(early.alsfrs[needed]))
    ok = np.isfinite(early.alsfrs)
    np.testing.assert_array_equal(early.alsfrs[ok], full.alsfrs[ok])
    np.testing.assert_array_equal(early.symptom_onset, full.symptom_onset)


def test_spread_rate_zero_confines_disease():
    res = simulate(TYPICAL.replace(spread_rate=0.0), months=60)
    others = [mndsim.REGIONS.index(r) for r in ("bulbar", "respiratory", "lower_limb")]
    assert np.allclose(res.mvc[0, :, others], 1.0)
    assert np.isnan(res.event_time[0])


def test_region_severity_can_spare_a_region():
    res = simulate(TYPICAL.replace(region_severity={"bulbar": 0.0}), months=80)
    assert np.allclose(res.mu_count[0, :, mndsim.REGIONS.index("bulbar")], 120)


def test_record_units():
    res = simulate(TYPICAL, months=12, record_units=True)
    assert res.units_healthy.shape == (1, len(res.time), 4, 120)
    total = res.units_healthy + res.units_denervating
    assert np.all(total <= 1 + 1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        Patient(lmn_rate=-1)
    with pytest.raises(ValueError):
        Patient(onset_region="toe")
    with pytest.raises(ValueError):
        Patient(max_enlargement=0.5)
    with pytest.raises(ValueError):
        simulate(TYPICAL, months=0)
    with pytest.raises(ValueError):
        Treatment(lmn_protection=1.5)
    with pytest.raises(ValueError):
        Treatment(start=10, stop=5)


def test_patient_round_trip():
    p = TYPICAL.replace(region_severity={"lower_limb": 0.5}, seed=3)
    q = Patient.from_dict(p.to_dict())
    assert q == p
    assert math.isinf(Patient(spread_rate=0).regional_onset_times()[0])
