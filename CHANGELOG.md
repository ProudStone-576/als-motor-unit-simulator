# Changelog

## 0.2.0 — 2026-11-15

The hidden phase, and a zero-install web app.

### Added
- **Plasma neurofilament light (NfL)** as a mechanistic read-out. It is released by degenerating lower and upper motor neuron axons (weighted by axon size and length), with age-dependent healthy levels and first-order clearance. Without tuning beyond the overall scale, the virtual population shows NfL doubling a median of 11 months before symptoms and correlating with progression rate (r ≈ 0.5).
- `mndsim.hidden_phase`:
  - `response_lag` (biomarker vs functional response to treatment, by start stage);
  - `trigger_strategies`, `MonitoringRule`, `false_trigger_rate` and `threshold_tradeoff` (NfL-triggered treatment in presymptomatic carriers).
- Trials:
  - an NfL endpoint (treated/control geometric mean ratio);
  - `min_nfl_fold` enrichment;
  - `run_trials(..., duration=)` and `power_by_duration` for endpoint fingerprints across trial lengths.
- Summary columns `nfl_at_diagnosis`, `nfl_fold_at_diagnosis` and `nfl_lead_time`.
- New figures (`plot_hidden_phase`, `plot_power_by_duration`, `plot_prevention`) and CLI commands `mndsim lag` and `mndsim prevention`.
- **Browser engine** `web/mndsim.js`: a dependency-free port that reproduces the Python engine to about 1e-9 (golden-file parity test in CI), with browser-sized cohorts and trials.
- **Web app** "Inside ALS" (`web/`):
  - a live simulated muscle biopsy that shows fibre-type grouping, matched quantitatively to the engine's denervation;
  - hidden-phase charts and a treatment-timing comparison;
  - a virtual trial designer running in a Web Worker.
- GitHub Pages deployment workflow.


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
