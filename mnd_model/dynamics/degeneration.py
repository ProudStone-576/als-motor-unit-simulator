def update_umn_degeneration(umn, dt=1.0):
    """
    Simple linear degeneration model
    """
    rate = umn.state.get("degeneration_rate", 0.01)
    umn.state["degeneration_burden"] += rate * dt
    umn.state["degeneration_burden"] = min(1.0, umn.state["degeneration_burden"])


def update_lmn_degeneration(lmn, input_signal, dt=1.0):
    """
    LMN degeneration accelerated by excitotoxic stress
    """
    base_rate = lmn.state.get("degeneration_rate", 0.015)

    stress_threshold = 0.8
    stress = max(0.0, input_signal - stress_threshold)

    excitotoxic_factor = lmn.state.get("excitotoxicity_factor", 0.0)

    total_rate = base_rate + excitotoxic_factor * stress
    lmn.state["degeneration_burden"] += total_rate * dt
    lmn.state["degeneration_burden"] = min(1.0, lmn.state["degeneration_burden"])
