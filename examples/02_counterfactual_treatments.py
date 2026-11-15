"""Counterfactuals: the same patient under different mechanisms and timings.

python examples/02_counterfactual_treatments.py
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import mndsim
from mndsim import plotting
from mndsim.phenotypes import TYPICAL

untreated = mndsim.simulate(TYPICAL, months=120)
onset = untreated.symptom_onset[0]
diagnosis = onset + TYPICAL.diagnostic_delay

rows = []
for name in ("neuroprotective", "anti_excitotoxic", "nmj_stabilizer", "sprouting_enhancer", "symptomatic"):
    starts = (("presymptomatic", onset - 6), ("at diagnosis", diagnosis), ("1 year later", diagnosis + 12))
    for label, start in starts:
        res = mndsim.simulate(TYPICAL, mndsim.TREATMENTS[name].at(start), months=120)
        rows.append(
            {
                "treatment": name,
                "start": label,
                "months gained to respiratory failure": res.event_time[0] - untreated.event_time[0],
                "functional score 2y after diagnosis": res.value_at("alsfrs", diagnosis + 24)[0],
            }
        )
table = pd.DataFrame(rows)
print(table.round(1).to_string(index=False))
untreated_2y = untreated.value_at("alsfrs", diagnosis + 24)[0]
print(f"\nuntreated functional score 2y after diagnosis: {untreated_2y:.1f}")

# A combination: protections compose multiplicatively.
combo = mndsim.TREATMENTS["neuroprotective"] + mndsim.TREATMENTS["anti_excitotoxic"]
res = mndsim.simulate(TYPICAL, combo.at(diagnosis), months=120)
print(f"combination gains {res.event_time[0] - untreated.event_time[0]:.1f} months")

# Your own mechanism: a hypothetical drug that halves UMN loss and stabilises junctions,
# taking 3 months to reach full effect and stopped after 2 years.
custom = mndsim.Treatment(name="my_compound", umn_protection=0.5, nmj_rescue=0.08, ramp=3.0)
res = mndsim.simulate(TYPICAL, custom.at(diagnosis, stop=diagnosis + 24), months=120)
print(f"custom compound gains {res.event_time[0] - untreated.event_time[0]:.1f} months")

runs = {"untreated": untreated}
for name in ("neuroprotective", "nmj_stabilizer", "symptomatic"):
    runs[name] = mndsim.simulate(TYPICAL, mndsim.TREATMENTS[name].at(diagnosis), months=120)
fig = plotting.plot_trajectories(runs, "alsfrs", marks={"diagnosis / start": TYPICAL.diagnostic_delay})
plotting.save(fig, "counterfactuals.png")
print("wrote counterfactuals.png")
