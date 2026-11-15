"""Quick start: simulate one virtual patient and look at what happens.

python examples/01_quickstart.py
"""

import matplotlib

matplotlib.use("Agg")

import mndsim
from mndsim import plotting
from mndsim.phenotypes import TYPICAL

# One patient, 6 years from biological disease onset.
result = mndsim.simulate(TYPICAL, months=72)
print(result)
print(result.summary().T)

# The signature of compensation: by the time weakness is noticed, about half
# of the motor units in the affected region are already gone.
s = result.summary().iloc[0]
print(
    f"\nSymptoms begin {s.symptom_onset_months:.0f} months after disease onset, "
    f"when {100 * s.mu_loss_at_symptom_onset:.0f}% of motor units are already lost "
    f"(CMAP still {100 * s.cmap_at_symptom_onset:.0f}% of normal)."
)

# Tidy table of every recorded series (one row per month).
df = result.to_frame()
print(df[["month", "alsfrs", "fvc", "mu_count_upper_limb", "mvc_upper_limb"]].iloc[::12])

# Figures.
plotting.save(result.plot(), "quickstart_overview.png")

stochastic = mndsim.simulate(TYPICAL, months=60, stochastic=True, seed=1, record_units=True)
plotting.save(plotting.plot_motor_units(stochastic), "quickstart_motor_units.png")
print("\nwrote quickstart_overview.png and quickstart_motor_units.png")
