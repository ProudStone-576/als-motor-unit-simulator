# Motor Unit Degeneration Simulator

A small, readable simulation of how motor neuron disease (such as ALS) could progressively weaken muscle output, modelled as a signal chain from brain to muscle. It runs synthetic patients through the chain, applies hypothetical treatments, and records how force declines over time.

> **Scope:** This is an exploratory modelling project, not a validated clinical model. The parameters are illustrative and the "patients" are synthetic. Do not use it to make medical claims or decisions.

## The model

A motor unit is modelled as a graph of nodes joined by edges that carry a signal, with optional noise and attenuation:

```
Cortex -> UMN -> LMN -> NMJ -> Muscle
```

| Node | Role |
|---|---|
| Cortex | Produces the drive signal (`drive_intensity`) |
| UMN | Upper motor neuron. Output is scaled by its health and degeneration burden |
| LMN | Lower motor neuron. Same, plus extra damage when the incoming signal is high (excitotoxic stress) |
| NMJ | Neuromuscular junction, with a transmission efficiency |
| Muscle | Converts the signal into force |

Each time step does two things: a forward pass computes the muscle force, then a degeneration update raises each neuron's `degeneration_burden` (0 to 1). UMN loss is linear. LMN loss has a base rate plus a term that grows with excitotoxic stress above a threshold.

## Features

- **Synthetic cohorts:** four phenotypes in `cohorts/phenotypes.py` (fast progressor, slow progressor, UMN-dominant, LMN-dominant), each defined by its degeneration rates and excitotoxicity factor
- **Treatments:** rate multipliers that switch on at a start time (`treatments/library.py`), for example a neuroprotective drug or an excitotoxicity blocker
- **Hybrid neural ODE simulator:** replaces the hand-written degeneration rule with a learnable function integrated step by step (`neural_ode/`, needs PyTorch)

## Project structure

```
mnd_model/
  core/          Node and edge definitions and their forward functions
  physiology/    Builds the motor-unit graph and runs a forward pass
  dynamics/      Degeneration rules and the time-stepping simulator
  cohorts/       Synthetic patients, phenotypes, and a cohort runner
  treatments/    Treatment definition and a small library
  neural_ode/    Learnable degeneration dynamics (PyTorch)
  notebooks/     Exploration notebook
```

## Getting started

Python 3.12 is recommended.

```bash
git clone https://github.com/ProudStone-576/motor-unit-degeneration-simulator.git
cd motor-unit-degeneration-simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run a synthetic patient from the repo root:

```python
from mnd_model.cohorts.phenotypes import FAST_PROGRESSOR
from mnd_model.cohorts.run_cohort import run_patient

history = run_patient(FAST_PROGRESSOR, timesteps=250)
print(history["force"][-1])   # muscle force at the final step
```

`history` also holds `umn_degeneration` and `lmn_degeneration` over time, ready to plot.

## Limitations

- Linear and threshold-based degeneration only; no real patient data was used to fit or check it
- Time steps are unitless
- No tests yet

## Related work

The author's published computational ALS research is at [c9orf72-als-proteostasis-study](https://github.com/ProudStone-576/c9orf72-als-proteostasis-study).
