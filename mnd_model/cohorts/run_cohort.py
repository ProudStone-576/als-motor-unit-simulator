from mnd_model.cohorts.instantiate import instantiate_patient
from mnd_model.dynamics.simulator import simulate

def run_patient(patient, timesteps=250):
    cortex, umn, lmn, nmj, muscle, edges = instantiate_patient(patient)

    history = simulate(
        cortex,
        umn,
        lmn,
        nmj,
        muscle,
        edges,
        timesteps=timesteps,
        dt=1.0
    )

    return history
