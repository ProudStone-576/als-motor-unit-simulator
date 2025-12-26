import torch
from mnd_model.physiology.motor_unit import forward_pass

def simulate_neural_ode(
    cortex, umn, lmn, nmj, muscle, edges,
    ode_func,
    timesteps=200,
    dt=1.0
):
    history = {
        "force": [],
        "umn_deg": [],
        "lmn_deg": []
    }

    # Initial degeneration state
    z = torch.tensor([
        umn.state["degeneration_burden"],
        lmn.state["degeneration_burden"],
        0.0,  # placeholder for umn signal
        0.0   # placeholder for lmn signal
    ])

    for t in range(timesteps):

        # Forward physiology
        force = forward_pass(cortex, umn, lmn, nmj, muscle, edges)

        # Update ODE input signals
        z[2] = umn.output_signal
        z[3] = lmn.output_signal

        # Integrate degeneration
        z = z + dt * ode_func(None, z)

        # Clamp
        z[0] = torch.clamp(z[0], 0, 1)
        z[1] = torch.clamp(z[1], 0, 1)

        # Write back
        umn.state["degeneration_burden"] = float(z[0])
        lmn.state["degeneration_burden"] = float(z[1])

        # Record
        history["force"].append(force)
        history["umn_deg"].append(float(z[0]))
        history["lmn_deg"].append(float(z[1]))

    return history
