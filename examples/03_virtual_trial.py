"""Design an in-silico trial: power, endpoints, enrichment.

python examples/03_virtual_trial.py        # ~1-2 minutes
"""

import matplotlib

matplotlib.use("Agg")

import mndsim
from mndsim import plotting

treatment = mndsim.TREATMENTS["anti_excitotoxic"]

# 1. A conventional 12-month trial in an unselected population.
design = mndsim.TrialDesign(n_per_arm=150, duration=12)
pool = mndsim.build_trial_pool(treatment, design, pool_size=2000, seed=1)
print(pool, f"- screen failure {100 * pool.screen_failure_rate:.0f}%")
results = mndsim.run_trials(pool, n_trials=400, seed=2)
print("\nPower with 150 per arm:")
print(results.power().round(3).to_string(index=False))

# 2. How many patients do we need?
curve = mndsim.power_curve(pool, [50, 100, 150, 200], n_trials=300, seed=3)
print(curve.pivot(index="n_per_arm", columns="endpoint", values="power").round(2))
plotting.save(plotting.plot_power_curve(curve), "power_curve.png")

# 3. Enrich for faster progressors (Delta-FS >= 0.7 points/month at enrolment).
enriched = design.replace(min_delta_fs=0.7)
pool_fast = mndsim.build_trial_pool(treatment, enriched, pool_size=2000, seed=1)
fast = mndsim.run_trials(pool_fast, n_trials=400, seed=2).power().set_index("endpoint").power
base = results.power().set_index("endpoint").power
print(f"\nEnrichment: screen failure {100 * pool_fast.screen_failure_rate:.0f}%")
for ep in ("alsfrs_slope", "survival", "cafs"):
    print(f"  {ep:14s} power {base[ep]:.2f} -> {fast[ep]:.2f}")

# 4. Individual treatment effects - only knowable in silico.
effects = pool.individual_effects()
print("\nPoints of function preserved at 12 months, by region of onset:")
print(effects.groupby("onset_region").alsfrs_preserved.describe()[["mean", "25%", "50%", "75%"]].round(2))

# 5. Placebo check: the false-positive rate should be close to alpha.
placebo = mndsim.build_trial_pool(mndsim.TREATMENTS["placebo"], design, pool_size=2000, seed=1)
fp = mndsim.run_trials(placebo, n_trials=400, seed=2).table
print(f"\nPlacebo: functional-slope p < 0.05 in {100 * (fp.alsfrs_slope_p < 0.05).mean():.1f}% of trials")
