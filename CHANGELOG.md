# Changelog

## 0.1.0 — 2026-11-15

A complete rewrite of the original prototype as an installable, tested package (`mndsim`).

### Added
- Motor-unit pool per region following Fuglevand et al. (1993): size principle, rate coding, force–frequency saturation.
- Disease engine:
  - healthy → denervating (dying-back) → lost unit states, with selective vulnerability by unit type;
  - collateral reinnervation with limited capacity, and irreversible loss of long-denervated fibres;
  - UMN loss reducing voluntary drive, and excitotoxic UMN → LMN coupling;
  - focal onset with contiguous spread between four regions;
  - mean-field and stochastic modes, vectorised across patients and processed in chunks.
- Clinical and electrophysiological read-outs: a 48-point functional score modelled on the ALSFRS-R structure, FVC, MUNE, CMAP, MUP size, unit enlargement, denervation, symptom onset and respiratory-failure events.
- Ten preset phenotypes, including PLS-like, PMA-like, flail-arm and SOD1-familial patients.
- Mechanism-based treatments with ramp and washout, combinations, and genotype targeting.
- Virtual population model and cohort simulation.
- In-silico randomised trials with potential outcomes, five endpoints (functional slope, log-rank survival, CAFS, MUNE, FVC), power curves and enrichment.
- Digital-twin fitting with confidence intervals and counterfactual forecasts.
- Global sensitivity analysis (Latin hypercube + PRCC) and one-at-a-time sweeps.
- Plotting module, `mndsim` command-line tool, Streamlit app, examples and model documentation.
- Test suite (65 tests, 97% coverage), and CI on Linux, macOS and Windows.

### Fixed (relative to the prototype)
- The treatment excitotoxicity multiplier compounded at every time step after treatment started.
- LMN excitotoxic stress was computed from the LMN's own output rather than its input.
- A degeneration rate of zero caused a division by zero.
- The neural-ODE module was never trained and modified tensors in place; it is replaced by mechanistic fitting (`fit_patient`).

### Removed
- The committed Windows conda environment (more than 800 files), a 17 MB archive duplicating the project, and compiled caches.
