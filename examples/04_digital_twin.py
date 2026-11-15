"""Fit a digital twin to a patient's visits and explore what-ifs.

The visits in ``examples/data/example_patient.csv`` are SYNTHETIC (generated
by this simulator with added rater noise); they are not from a real person.

    python examples/04_digital_twin.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import mndsim
from mndsim import plotting

data = pd.read_csv(pathlib.Path(__file__).parent / "data" / "example_patient.csv")
print(data)

fit = mndsim.fit_patient(
    data.months_since_onset,
    data.alsfrs,
    fvc=data.fvc,
    onset_region="bulbar",
    parameters=("lmn_rate", "spread_rate"),
)
print(f"\nconverged: {fit.success}; RMSE {fit.rmse_alsfrs:.2f} points")
print(fit.confidence_intervals().round(4).to_string(index=False))

untreated = fit.simulate(months_after_onset=60)
onset = fit.symptom_onset
print(f"\npredicted respiratory failure: {untreated.event_time[0] - onset:.1f} months after symptom onset")
for start in (6.0, 12.0, 18.0):
    cf = fit.counterfactual(mndsim.TREATMENTS["neuroprotective"], start_after_onset=start)
    print(f"  neuroprotection from month {start:>4.0f}: {cf.event_time[0] - onset:.1f} months")

runs = {
    "twin, untreated": untreated,
    "neuroprotection from month 6": fit.counterfactual(mndsim.TREATMENTS["neuroprotective"], 6.0),
}
fig = plotting.plot_trajectories(runs, "alsfrs", marks={"start": 6.0})
fig.axes[0].scatter(data.months_since_onset, data.alsfrs, s=18, color="#0b0b0b", zorder=5, label="visits")
fig.axes[0].legend(frameon=False, fontsize=8.5)
plotting.save(fig, "digital_twin.png")
print("wrote digital_twin.png")
