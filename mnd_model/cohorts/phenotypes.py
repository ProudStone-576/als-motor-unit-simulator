from mnd_model.cohorts.patient import SyntheticPatient

FAST_PROGRESSOR = SyntheticPatient(
    umn_degeneration_rate=0.006,
    lmn_degeneration_rate=0.02,
    excitotoxicity_factor=0.8,
    label="fast_progressor"
)

SLOW_PROGRESSOR = SyntheticPatient(
    umn_degeneration_rate=0.002,
    lmn_degeneration_rate=0.006,
    excitotoxicity_factor=0.3,
    label="slow_progressor"
)

UMN_DOMINANT = SyntheticPatient(
    umn_degeneration_rate=0.01,
    lmn_degeneration_rate=0.004,
    excitotoxicity_factor=0.2,
    label="umn_dominant"
)

LMN_DOMINANT = SyntheticPatient(
    umn_degeneration_rate=0.002,
    lmn_degeneration_rate=0.015,
    excitotoxicity_factor=0.7,
    label="lmn_dominant"
)
