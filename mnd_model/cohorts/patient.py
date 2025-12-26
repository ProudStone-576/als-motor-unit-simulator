class SyntheticPatient:
    def __init__(
        self,
        umn_degeneration_rate,
        lmn_degeneration_rate,
        excitotoxicity_factor,
        cortical_drive=1.0,
        label="generic"
    ):
        self.umn_degeneration_rate = umn_degeneration_rate
        self.lmn_degeneration_rate = lmn_degeneration_rate
        self.excitotoxicity_factor = excitotoxicity_factor
        self.cortical_drive = cortical_drive
        self.label = label
