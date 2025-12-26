class Treatment:
    def __init__(
        self,
        umn_rate_multiplier=1.0,
        lmn_rate_multiplier=1.0,
        excitotoxicity_multiplier=1.0,
        start_time=0,
        label="generic"
    ):
        self.umn_rate_multiplier = umn_rate_multiplier
        self.lmn_rate_multiplier = lmn_rate_multiplier
        self.excitotoxicity_multiplier = excitotoxicity_multiplier
        self.start_time = start_time
        self.label = label
