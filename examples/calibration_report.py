"""Print the calibration of the default model against population-level targets.

    python examples/calibration_report.py

The numbers in the README's calibration table come from this script.
"""

import numpy as np

import mndsim
from mndsim.metrics import median_survival
from mndsim.phenotypes import TYPICAL

HORIZON = 240.0


def km_median(summary, mask=None):
    surv = summary.survival_from_symptom_onset.to_numpy()
    onset = summary.symptom_onset_months.to_numpy()
    if mask is not None:
        surv, onset = surv[mask], onset[mask]
    observed = np.isfinite(surv)
    follow = np.where(observed, surv, HORIZON - onset)
    ok = np.isfinite(follow)
    return median_survival(follow[ok], observed[ok]), follow[ok], observed[ok]


def main() -> None:
    typical = mndsim.simulate(TYPICAL, months=120).summary().iloc[0]
    print("Typical patient (upper-limb onset)")
    print(f"  survival from symptom onset          {typical.survival_from_symptom_onset:6.1f} months")
    print(f"  functional slope, first year          {typical.alsfrs_slope_first_year:6.2f} points/month")
    print(f"  functional score at diagnosis         {typical.alsfrs_at_diagnosis:6.1f} / 48")
    print(f"  motor units lost at symptom onset     {100 * typical.mu_loss_at_symptom_onset:6.0f} %")
    print(f"  CMAP at symptom onset                 {100 * typical.cmap_at_symptom_onset:6.0f} % of normal")

    patients = mndsim.sample_cohort(3000, seed=0)
    s = mndsim.simulate(patients, months=HORIZON, dt=0.5).summary()
    med, follow, _ = km_median(s)
    print("\nVirtual population (n=3000, default PopulationModel)")
    print(f"  median survival from symptom onset    {med:6.1f} months")
    print(f"  alive at 5 years                      {100 * np.mean(follow > 60):6.1f} %")
    print(f"  alive at 10 years                     {100 * np.mean(follow > 120):6.1f} %")
    for region in ("bulbar", "upper_limb", "lower_limb"):
        m, *_ = km_median(s, (s.onset_region == region).to_numpy())
        print(f"  median survival, {region:11s} onset   {m:6.1f} months")
    q = np.nanpercentile(s.alsfrs_slope_first_year, [10, 50, 90])
    print(f"  first-year slope 10/50/90th pct       {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} points/month")
    print(f"  median Delta-FS at diagnosis          {np.nanmedian(s.delta_fs):6.2f} points/month")
    print(f"  median score at diagnosis             {np.nanmedian(s.alsfrs_at_diagnosis):6.1f} / 48")
    print(f"  median motor units lost at onset      {100 * np.nanmedian(s.mu_loss_at_symptom_onset):6.0f} %")
    rho = np.corrcoef(np.log(s.delta_fs), np.log(follow))[0, 1]
    print(f"  corr(log Delta-FS, log survival)      {rho:6.2f}")


if __name__ == "__main__":
    main()
