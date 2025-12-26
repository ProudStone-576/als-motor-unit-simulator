from mnd_model.treatments.treatments import Treatment

NEUROPROTECTIVE = Treatment(
    umn_rate_multiplier=0.7,
    lmn_rate_multiplier=0.6,
    excitotoxicity_multiplier=0.8,
    start_time=50,
    label="neuroprotective"
)

EXCITOTOXIC_BLOCKER = Treatment(
    umn_rate_multiplier=1.0,
    lmn_rate_multiplier=1.0,
    excitotoxicity_multiplier=0.4,
    start_time=30,
    label="excitotoxic_blocker"
)
