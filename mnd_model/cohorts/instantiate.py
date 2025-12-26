from mnd_model.physiology.motor_unit import build_motor_unit

def instantiate_patient(patient):
    cortex, umn, lmn, nmj, muscle, edges = build_motor_unit()

    cortex.state["drive_intensity"] = patient.cortical_drive

    umn.state["degeneration_rate"] = patient.umn_degeneration_rate
    lmn.state["degeneration_rate"] = patient.lmn_degeneration_rate
    lmn.state["excitotoxicity_factor"] = patient.excitotoxicity_factor

    return cortex, umn, lmn, nmj, muscle, edges
