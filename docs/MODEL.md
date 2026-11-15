# Model description

This document specifies the model implemented in `mndsim` v0.2. It covers the state variables, equations, default parameters (and where each comes from), read-outs and numerical method. Code references point to `src/mndsim/`.

Contents:

1. [Overview](#1-overview)
2. [Motor-unit pool](#2-motor-unit-pool-poolpy)
3. [Disease dynamics](#3-disease-dynamics-enginepy)
4. [Treatments](#4-treatments-treatmentpy)
5. [Read-outs](#5-read-outs)
6. [Numerical method](#6-numerical-method)
7. [Parameters](#7-parameters)
8. [Virtual population](#8-virtual-population-cohortpy)
9. [In-silico trials](#9-in-silico-trials-trialpy)
10. [Fitting and sensitivity](#10-fitting-and-sensitivity)
11. [Neurofilament light and the hidden phase](#11-neurofilament-light-and-the-hidden-phase)
12. [Browser engine](#12-browser-engine-webmndsimjs)
13. [References](#13-references)

---

## 1. Overview

```
              upper motor neurons U_r ──── maximal drive E_r ───────────────┐
                    │  (excitotoxic drive X_r)                              │
                    ▼                                                       ▼
 healthy ──λ_i──▶ denervating ──1/τ_G──▶ lost         force F_r = Σ_i fibres_i · φ_i(E_r)
    ▲                │   │                │
    └──NMJ rescue────┘   │ terminal       │ remaining fibres
                         │ retraction     ▼
                         └──────────▶ orphaned fibres D_r ──τ_atr──▶ irreversibly lost L_r
                                           │
                     collateral sprouting  ▼
                    (healthy units adopt orphans, up to a maximum enlargement)
```

The body has four regions *r* ∈ {bulbar, respiratory, upper limb, lower limb}. Each has a pool of *N* = 120 lower motor units *i* and a population of upper motor neurons. Time *t* is in **months since biological disease onset** in the first region affected.

## 2. Motor-unit pool (`pool.py`)

The structure follows Fuglevand, Winter & Patla (1993). With *i* = 1…*N* ordered from smallest to largest:

| Property | Formula | Default |
|---|---|---|
| Peak twitch force | $P_i = \exp(i \ln R_P / N)$ | $R_P = 100$ |
| Recruitment threshold | $\theta_i = \exp(i \ln R_R / N)$ | $R_R = 30$ |
| Peak firing rate | $\mathrm{PFR}_i = \mathrm{PFR}_1 - \Delta_{\mathrm{PFR}}\,\theta_i/\theta_N$ | 35 Hz, Δ = 10 Hz |
| Contraction time | $T_i = T_L (P_i/P_1)^{-\ln R_T/\ln R_P}$ | $T_L$ = 90 ms, $R_T$ = 3 |
| Fibres (fraction of muscle) | $f_i = P_i / \sum_j P_j$ | sums to 1 |
| Type | S / FR / FF by size rank | 50% / 30% / 20% of units |

**Rate coding.** At excitation *E*, unit *i* fires at $r_i(E) = \min(g_e (E-\theta_i) + \mathrm{MFR},\ \mathrm{PFR}_i)$ if $E \ge \theta_i$, else 0, with MFR = 8 Hz and $g_e$ = 1 Hz per unit of excitation.

**Force.** Each unit's mean force per fibre is a function of the normalised rate $x = T_i r_i$:

$$
\varphi(x) = \begin{cases} e\,x & x \le 0.4 \\ e\cdot 0.4\,\dfrac{1-\exp(-2x^3)}{1-\exp(-2\cdot0.4^3)} & x > 0.4 \end{cases}
$$

Below 0.4, twitches sum linearly. Above it, force saturates towards fused tetanus. The function is continuous at 0.4. Pool force is

$$F(E) = \frac{\sum_i n_i^{\mathrm{eff}}\ \varphi(T_i r_i(E))}{\sum_i f_i\ \varphi(T_i r_i(E_{\max}))}$$

so a healthy pool at maximal excitation $E_{\max}$ produces exactly 1. Here $E_{\max} = \theta_N + (\mathrm{PFR}_N - \mathrm{MFR})/g_e$ and $n_i^{\mathrm{eff}}$ is the unit's effective fibre count (§3). The engine evaluates φ from a 2,049-point table. Recruitment makes per-unit force discontinuous, so the worst-case interpolation error is about 0.2% of healthy force.

## 3. Disease dynamics (`engine.py`)

### 3.1 Spread

Each region has an effective neuraxis position *p_r* (bulbar 0, respiratory 0.5, upper limb 1.0, lower limb 1.7). Respiratory sits between the bulbar and brachial pools because its function depends on phrenic (C3–C5) and intercostal pools. The lumbar position is compressed so lower-limb onset reaches other regions on a clinically plausible timescale. Disease reaches region *r* at

$$t_r = \frac{|p_r - p_{\mathrm{onset}}|}{v}$$

where *v* is the patient's `spread_rate` (Ravits & La Spada 2009). Activity then ramps up as $a_r(t) = 1 - e^{-(t-t_r)/\tau_a}$ for $t \ge t_r$, with $\tau_a$ = 2 months.

### 3.2 Upper motor neurons

The surviving fraction $U_r$ decays as

$$\frac{dU_r}{dt} = -\,\mu_U\, s_r\, a_r(t)\, M_U(t)\, U_r$$

with patient rate $\mu_U$ (`umn_rate`), regional severity $s_r$ and treatment multiplier $M_U$. Maximal excitation of the pool falls with UMN loss but not to zero, because non-corticospinal pathways remain:

$$E_r = E_{\max}\,[\rho + (1-\rho)\,U_r], \qquad \rho = 0.5 \ (\texttt{residual\_drive})$$

### 3.3 Lower motor neurons

Each unit is **healthy (H)**, **denervating (G)** or **lost (X)**. The H → G hazard is

$$\lambda_{ri}(t) = \lambda\, s_r\, a_r(t)\, \big(1 + \kappa_{\mathrm{exc}}\, U_r\, M_X(t)\big)\, M_L(t)\, v_i$$

where:

- *λ* is `lmn_rate`.
- *κ_exc* is `excitotoxicity`: the glutamatergic drive of surviving UMNs ("dying forward"; Eisen et al. 1992, Vucic et al. 2008).
- $v_i$ is the unit's vulnerability: a type multiplier (S 0.35, FR 1.0, FF 2.2; FF units are lost first, Frey et al. 2000, Pun et al. 2006) times a mean-one log-normal factor (SD 0.35) drawn per unit and per patient.

Denervating units leave G at rate $1/\tau_G + \rho_{\mathrm{rescue}}(t)$. They die (G → X) with rate $1/\tau_G$, where $\tau_G$ = 4 months is the dying-back duration (Fischer et al. 2004). With an NMJ-stabilising treatment they return to H at rate $\rho_{\mathrm{rescue}}$.

### 3.4 Fibres

Let $n^H_{ri}$ and $n^G_{ri}$ be the fibres innervated by unit *i* while it is healthy or denervating; $D_r$ the orphaned fibres that are still viable; and $L_r$ the fibres lost irreversibly. Initially $n^H = f$, $n^G = D = L = 0$. Fibres move with their unit between H and G. In addition:

- **Terminal retraction.** A denervating unit sheds fibres to $D_r$ at rate $1/\tau_{\mathrm{ret}}$ ($\tau_{\mathrm{ret}}$ = 3 months).
- **Death.** A lost unit's remaining fibres join $D_r$.
- **Collateral sprouting.** Healthy units adopt orphans. Each unit's spare capacity is $c_i = \max(0,\ \epsilon_{\max} M_C f_i\, p^H_i - n^H_i)$, with $\epsilon_{\max}$ = `max_enlargement` (default 5) and $p^H_i$ the probability the unit is healthy. Units are weighted by type competence $k_i$ (S 1.0, FR 0.7, FF 0.3; Frey et al. 2000), $w_i = k_i c_i$. The total adoption rate is $\sigma M_S \sum_i w_i$, with σ = `sprouting_rate` (default 2/month). It is shared in proportion to $w_i$, so reinnervation slows as capacity is used up.
- **Atrophy.** Orphans not reinnervated in time degenerate: $D_r \to L_r$ at rate $1/\tau_{\mathrm{atr}}$ ($\tau_{\mathrm{atr}}$ = 15 months).

Fibres are conserved: $\sum_i (n^H_i + n^G_i) + D_r + L_r = 1$. A test checks this.

Effective fibres for force are $n^{\mathrm{eff}}_i = n^H_i + \gamma\, n^G_i$, where γ = 0.6 is the transmission reliability of failing junctions.

### 3.5 Mean-field and stochastic modes

In **mean-field** mode, the state is the probability of each unit being in each state, with expected fibre counts, so one run gives the expected trajectory. In **stochastic** mode, each unit's state is sampled every step with the same transition probabilities. Fibre flows are then all-or-nothing, which produces step-wise MUNE decrements. Averaging many stochastic runs reproduces the mean-field trajectory (tested).

## 4. Treatments (`treatment.py`)

Each treatment *k* has an efficacy profile $e_k(t) \in [0,1]$. It is 0 before `start`, rises as $1 - e^{-(t-t_{\mathrm{start}})/\tau_{\mathrm{ramp}}}$, and decays as $e^{-(t-t_{\mathrm{stop}})/\tau_{\mathrm{wash}}}$ after `stop`. Genotype-targeted treatments act only in matching patients. Effects combine across treatments as follows:

| Multiplier | Combination | Treatment field |
|---|---|---|
| $M_L$ (LMN hazard) | $\prod_k (1 - e_k\,\mathrm{lmn\_protection}_k)$ | `lmn_protection` |
| $M_U$ (UMN loss) | $\prod_k (1 - e_k\,\mathrm{umn\_protection}_k)$ | `umn_protection` |
| $M_X$ (excitotoxic drive) | $\prod_k (1 - e_k\,\mathrm{anti\_excitotoxic}_k)$ | `anti_excitotoxic` |
| $\rho_{\mathrm{rescue}}$ (G → H) | $\sum_k e_k\,\mathrm{nmj\_rescue}_k$ | `nmj_rescue` |
| $M_S$ (sprouting rate) | $\prod_k (1 + e_k\,\mathrm{sprouting\_boost}_k)$ | `sprouting_boost` |
| $M_C$ (maximum enlargement) | $\prod_k (1 + e_k\,\mathrm{capacity\_boost}_k)$ | `capacity_boost` |
| force | $\prod_k (1 + e_k\,\mathrm{symptomatic}_k)$ | `symptomatic` |

Library effect sizes are archetypes for exploring hypotheses. They are not estimates for any real drug.

## 5. Read-outs

| Read-out | Definition |
|---|---|
| `mvc` | $F(E_r)$, maximal voluntary force per region (1 = healthy) |
| `mu_count` (MUNE) | $\sum_i (p^H_i + p^G_i)$ |
| `cmap` | $\sum_i n^{\mathrm{eff}}_i$, CMAP amplitude relative to baseline |
| `mup_size` | `cmap` × N / `mu_count`, mean motor-unit potential size relative to the healthy mean |
| `enlargement` | $\sum_i (n^H_i+n^G_i) / \sum_i (p^H_i+p^G_i) f_i$, surviving units' size relative to their own original size |
| `denervated` | $D_r$, viable orphaned fibres (fibrillation potentials on EMG) |
| `type_survival` | surviving fraction of S, FR and FF units |
| `items`, `alsfrs` | 12 items scored 0–4 (§5.1), total 0–48 |
| `fvc` | 100 × respiratory `mvc` (% predicted) |
| `nfl` | plasma neurofilament light, pg/mL on an illustrative scale (§11) |
| `symptom_onset` | first time any region's `mvc` < 0.80 (interpolated within a step) |
| `event_time` | first time `fvc` < 30%: "death or permanent assisted ventilation" |

### 5.1 Functional score

The score follows the structure of the ALSFRS-R (Cedarbaum et al. 1999): bulbar, fine motor, gross motor and respiratory domains, each with three items scored 0–4. It is **not** the clinical instrument. Each item is a rescaled logistic function of the strength *c* of the region (or regions) that drive it:

$$\mathrm{item}(c) = 4\,\frac{\ell\big((c - c_{50})/w\big) - \ell(-c_{50}/w)}{\ell\big((1-c_{50})/w\big) - \ell(-c_{50}/w)}, \qquad \ell(z) = \frac{1}{1+e^{-z}}$$

This maps *c* = 0 to 0 and *c* = 1 to 4, with *w* = 0.15. The half-loss points $c_{50}$ are listed below. Items with a higher $c_{50}$ (for example, climbing stairs) are lost earlier.

| Domain | Items ($c_{50}$) |
|---|---|
| Bulbar (bulbar region) | speech 0.55, salivation 0.40, swallowing 0.45 |
| Fine motor (upper limb) | handwriting 0.60, cutting food 0.50, dressing & hygiene 0.45 |
| Gross motor | turning in bed 0.40 (mean of upper and lower limb), walking 0.50, climbing stairs 0.62 (lower limb) |
| Respiratory (respiratory region) | dyspnoea 0.60, orthopnoea 0.50, respiratory insufficiency 0.35 |

For trials, `observe_items` adds Gaussian rater noise (SD 0.3 per item), then rounds and clips to the 0–4 ordinal scale.

## 6. Numerical method

State variables are arrays shaped (patients, regions, units). Every step uses exact exponential transition probabilities for the rates at the start of the step. Examples are $1 - e^{-\lambda\,dt}$ for H → G, and competing-risk splitting for G → {X, H}. This keeps the scheme unconditionally stable and accurate at the default dt = 0.25 month: event times at dt = 0.5 and 0.125 agree to within a month (tested). Event times are linearly interpolated within a step. Large batches run in chunks of 256 patients for cache efficiency, and results are identical whatever the chunk size. Optional per-patient early stopping (`stop_time`, `stop_after_onset`) speeds up trial simulations.

## 7. Parameters

### Patient (`Patient`)

| Name | Default (typical) | Meaning |
|---|---|---|
| `onset_region` | upper_limb | region of first degeneration |
| `lmn_rate` | 0.08 /month | baseline H → G hazard of an FR unit |
| `umn_rate` | 0.01 /month | UMN loss rate |
| `excitotoxicity` | 0.5 | LMN hazard amplification by UMN drive |
| `spread_rate` | 0.04 /month | neuraxis units per month |
| `sprouting_rate` | 2.0 /month | collateral reinnervation rate constant |
| `max_enlargement` | 5× | maximum size of a surviving unit |
| `region_severity` | 1, 1, 1, 1 | per-region rate multipliers |
| `genotype` | sporadic | label used by genotype-targeted treatments |
| `diagnostic_delay` | 12 months | symptom onset to diagnosis (ΔFS, trial enrolment) |

### Physiology (`PhysiologyConfig`)

| Name | Default | Provenance |
|---|---|---|
| `n_units` | 120 | Fuglevand et al. 1993 |
| `twitch_force_range`, `recruitment_range` | 100, 30 | Fuglevand et al. 1993 |
| `min_firing_rate`, `peak_firing_rate_first`, `peak_firing_rate_diff` | 8, 35, 10 Hz | Fuglevand et al. 1993 |
| `longest_contraction_time`, `contraction_time_range` | 90 ms, 3 | Fuglevand et al. 1993 |
| `type_fractions` | 0.5 / 0.3 / 0.2 | illustrative |
| `type_vulnerability` | 0.35 / 1.0 / 2.2 | ordering from Frey 2000 and Pun 2006; magnitudes illustrative |
| `unit_vulnerability_sd` | 0.35 | illustrative |
| `type_sprouting` | 1.0 / 0.7 / 0.3 | ordering from Frey 2000; magnitudes illustrative |
| `dying_back_duration` | 4 months | illustrative (dying-back: Fischer 2004) |
| `terminal_retraction_time` | 3 months | illustrative |
| `denervating_transmission` | 0.6 | illustrative |
| `denervated_fiber_survival` | 15 months | illustrative |
| `residual_drive` | 0.5 | illustrative |
| `regional_ramp_time` | 2 months | illustrative |
| `symptom_threshold` | 0.80 of healthy strength | illustrative |
| `respiratory_failure_threshold` | 0.30 (FVC 30%) | illustrative |

"Illustrative" means the value was chosen, together with the patient defaults, to reproduce the population-level targets in the README's calibration table. It is not a measurement. Better-sourced values are the most valuable contribution you can make.

## 8. Virtual population (`cohort.py`)

`PopulationModel` multiplies the typical patient's rate parameters by log-normal factors. The log-SDs are: LMN rate 0.6, UMN rate 0.7, excitotoxicity 0.5, spread 0.6, sprouting 0.3, enlargement 0.12. A shared "aggressiveness" factor correlates LMN rate and spread (log-scale correlation 0.6).

Other population settings:

- **Onset region:** upper limb 36%, lower limb 34%, bulbar 28%, respiratory 2%. Bulbar onset adds ×1.25 to LMN rate and ×1.2 to spread, and is 5 years older on average.
- **Genotype:** sporadic 90%, C9orf72 8%, SOD1 2% (labels only).
- **Age:** normal(61, 11).
- **Diagnostic delay:** log-normal with median 11 months and log-SD 0.5.

All of these are editable defaults.

## 9. In-silico trials (`trial.py`)

1. **Screening.** Patients are drawn from the population and simulated untreated. Each is enrolled at symptom onset + diagnostic delay + U(0, 3) months. Eligibility is checked at enrolment: alive, symptom duration ≤ 24 months, FVC ≥ 70%, and optionally a minimum ΔFS or a specific genotype.
2. **Potential outcomes.** Eligible patients are simulated again with treatment starting at enrolment. Both arms are identical up to enrolment (tested).
3. **Replicate trials.**
   - Draw 2*n* patients without replacement and randomise 1:1.
   - Apply exponential dropout and follow-up censoring.
   - Add rater noise to items, log-normal noise to MUNE (CV 0.15) and additive noise to FVC (SD 5).
4. **Analyses.**
   - Per-patient OLS slopes of the functional score, log-MUNE and FVC, compared with Welch's t-test.
   - Survival compared with a log-rank test, with Peto's one-step hazard ratio $\exp((O-E)/V)$.
   - CAFS (Berry et al. 2013): pairwise comparisons on survival, then functional slope, with a Mann–Whitney test on the scores.
5. **Power.** The fraction of replicates significant at α (two-sided) *and* in favour of treatment.

Replicates resample a finite pool, so power estimates are conditional on that pool. This matters most for rare endpoints such as deaths in short trials, so check several seeds.

## 10. Fitting and sensitivity

**`fit_patient`** fits selected log-parameters to observed functional scores, FVC and/or MUNE at known months since symptom onset. Model and data are aligned on the model's emergent symptom onset. The fit runs a vectorised grid search, then bounded `scipy.optimize.least_squares` from the three best grid points. Residuals are standardised by measurement SDs (functional score 2, FVC 8, log-MUNE 0.2). Approximate standard errors come from the Jacobian, $\mathrm{cov} \approx s^2 (J^\top J)^{-1}$.

**`global_sensitivity`** draws a Latin hypercube over patient parameters (log-uniform) and ranks each one's influence on each output with partial rank correlation coefficients (Marino et al. 2008). Outcomes not reached within the horizon are ranked as largest.

## 11. Neurofilament light and the hidden phase

### 11.1 NfL release and clearance (`engine.py`)

Neurofilament light chain is a structural protein of large axons. It enters blood when axons degenerate. In the model, release is proportional to the axonal material lost per unit time:

- **Lower motor neurons.** Unit *i* contributes $a_i L_r$ "axon equivalents". Content scales with unit size, $a_i \propto f_i^{\,0.25}$ normalised to mean 1, and with the region's relative axon length $L_r$ (bulbar 0.25, respiratory 0.5, upper limb 0.8, lower limb 1.0). A fraction $\phi = 0.3$ is released when the unit starts dying back (H → G, distal axon) and the rest when it is lost (G → X). Rescued units release nothing more.
- **Upper motor neurons.** A region's whole UMN population holds 150 equivalents, released in proportion to UMN loss.

Plasma concentration follows first-order kinetics with healthy production and clearance:

$$\frac{dC}{dt} = k\,(C_0(\text{age}) - C) + \kappa\, S(t), \qquad C_0 = 10\,e^{0.022(\text{age}-60)}\ \text{pg/mL}, \qquad k = \ln 2 / t_{1/2}$$

*S* is the release rate in axon equivalents per month, κ = 3.3 pg/mL per equivalent per month, and $t_{1/2}$ = 1 month. The update is the exact solution over each step. The absolute scale is illustrative, so read NfL as a multiple of the healthy level for age.

With these settings the virtual population (not tuned for NfL beyond the overall scale) shows:

- a median NfL at diagnosis of about 4.5× healthy;
- a correlation with the progression rate ΔFS of r ≈ 0.5;
- a slow decline of about 13% over the year after diagnosis;
- NfL that **doubles a median of 11 months before the first symptom** (IQR 5–17 months). This is consistent with prospective studies of gene carriers that report NfL rising in the months to about a year before clinical onset.

### 11.2 Analyses (`hidden_phase.py`)

- **`response_lag`** compares one patient with and without a treatment started at canonical stages (6 months before symptoms, at symptom onset, at diagnosis, one year later). It reports months until NfL first falls 15% below untreated, months until function is first 1 point better, the maximum NfL reduction, and months gained before respiratory failure.
- **`power_by_duration`** (`trial.py`) builds one trial pool with the longest follow-up, then re-analyses replicate trials truncated at shorter lengths. Patterns of power across endpoints and durations act as mechanism *fingerprints*:
  - Disease-modifying treatments are detected first by NfL, then by function.
  - Symptomatic treatments are detected early by function and never by NfL.
- **`trigger_strategies`** samples presymptomatic gene carriers (`carrier_population`).
  - Monitoring starts 2 years before biological onset, with blood draws every 3 months.
  - Each draw has a log-normal visit-to-visit CV of 0.15, and people's healthy NfL varies by CV 0.3.
  - A `MonitoringRule` fires when measured NfL exceeds a threshold times a personal baseline (mean of the first two draws) or the population reference. By default, two consecutive draws must be above threshold.
  - Treatment started at the trigger is compared with starting at symptom onset or at diagnosis. Pre-treatment trajectories are identical across strategies by construction.
- **`false_trigger_rate`** and **`threshold_tradeoff`** estimate how often a rule fires in carriers who never convert. They include the slow age-related rise in healthy NfL.

## 12. Browser engine (`web/mndsim.js`)

The web app runs a dependency-free JavaScript port of the engine. It simulates one patient at a time, which allows per-patient early stopping, and uses the same equations, operation order and 2,049-point force table as the Python engine. `web/tests/make_golden.py` records trajectories from the Python engine, including each patient's unit vulnerabilities. `web/tests/parity.test.mjs` requires the JavaScript engine to reproduce them; the observed worst relative difference is about 1e-9. Random numbers differ between the two engines (seeded mulberry32 in JavaScript), so stochastic runs and sampled cohorts agree statistically, not draw for draw.

## 13. References

- Benatar M, Wuu J, Andersen PM, Lombardi V, Malaspina A. Neurofilament light: a candidate biomarker of presymptomatic amyotrophic lateral sclerosis and phenoconversion. *Ann Neurol* 2018;84:130–139.
- Berry JD, Miller R, Moore DH, et al. The Combined Assessment of Function and Survival (CAFS): a new endpoint for ALS clinical trials. *Amyotroph Lateral Scler Frontotemporal Degener* 2013;14:162–168.
- Cedarbaum JM, Stambler N, Malta E, et al. The ALSFRS-R: a revised ALS functional rating scale that incorporates assessments of respiratory function. *J Neurol Sci* 1999;169:13–21.
- Eisen A, Kim S, Pant B. Amyotrophic lateral sclerosis (ALS): a phylogenetic disease of the corticomotoneuron? *Muscle Nerve* 1992;15:219–224.
- Fischer LR, Culver DG, Tennant P, et al. Amyotrophic lateral sclerosis is a distal axonopathy: evidence in mice and man. *Exp Neurol* 2004;185:232–240.
- Frey D, Schneider C, Xu L, Borg J, Spooren W, Caroni P. Early and selective loss of neuromuscular synapse subtypes with low sprouting competence in motoneuron diseases. *J Neurosci* 2000;20:2534–2542.
- Fuglevand AJ, Winter DA, Patla AE. Models of recruitment and rate coding organization in motor-unit pools. *J Neurophysiol* 1993;70:2470–2488.
- Henneman E. Relation between size of neurons and their susceptibility to discharge. *Science* 1957;126:1345–1347.
- Khalil M, Teunissen CE, Otto M, et al. Neurofilaments as biomarkers in neurological disorders. *Nat Rev Neurol* 2018;14:577–589.
- Kimura F, Fujimura C, Ishida S, et al. Progression rate of ALSFRS-R at time of diagnosis predicts survival time in ALS. *Neurology* 2006;66:265–267.
- Marino S, Hogue IB, Ray CJ, Kirschner DE. A methodology for performing global uncertainty and sensitivity analysis in systems biology. *J Theor Biol* 2008;254:178–196.
- Miller TM, Cudkowicz ME, Genge A, et al. Trial of antisense oligonucleotide tofersen for SOD1 ALS. *N Engl J Med* 2022;387:1099–1110.
- Pun S, Santos AF, Saxena S, Xu L, Caroni P. Selective vulnerability and pruning of phasic motoneuron axons in motoneuron disease alleviated by CNTF. *Nat Neurosci* 2006;9:408–419.
- Ravits JM, La Spada AR. ALS motor phenotype heterogeneity, focality, and spread: deconstructing motor neuron degeneration. *Neurology* 2009;73:805–811.
- Vucic S, Nicholson GA, Kiernan MC. Cortical hyperexcitability may precede the onset of familial amyotrophic lateral sclerosis. *Brain* 2008;131:1540–1550.
