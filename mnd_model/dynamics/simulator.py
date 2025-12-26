from mnd_model.physiology.motor_unit import forward_pass
from mnd_model.dynamics.degeneration import (
    update_umn_degeneration,
    update_lmn_degeneration
)

def simulate(
    cortex,
    umn,
    lmn,
    nmj,
    muscle,
    edges,
    timesteps=100,
    dt=1.0,
    treatment=None
):
    history = {
        "force": [],
        "umn_degeneration": [],
        "lmn_degeneration": []
    }

    for t in range(timesteps):

        force = forward_pass(cortex, umn, lmn, nmj, muscle, edges)

        history["force"].append(force)
        history["umn_degeneration"].append(umn.state["degeneration_burden"])
        history["lmn_degeneration"].append(lmn.state["degeneration_burden"])

        # Apply treatment modifiers
        if treatment and t >= treatment.start_time:
            umn_rate = umn.state["degeneration_rate"] * treatment.umn_rate_multiplier
            lmn_rate = lmn.state["degeneration_rate"] * treatment.lmn_rate_multiplier
            lmn.state["excitotoxicity_factor"] *= treatment.excitotoxicity_multiplier
        else:
            umn_rate = umn.state["degeneration_rate"]
            lmn_rate = lmn.state["degeneration_rate"]

        # Progress disease
        update_umn_degeneration(umn, dt * umn_rate / umn.state["degeneration_rate"])
        update_lmn_degeneration(lmn, lmn.output_signal, dt * lmn_rate / lmn.state["degeneration_rate"])

    return history
