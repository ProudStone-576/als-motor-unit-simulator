# mndsim — motor unit degeneration simulator

[![CI](https://github.com/ProudStone-576/motor-unit-degeneration-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/ProudStone-576/motor-unit-degeneration-simulator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A mechanistic, open simulator of motor neuron disease (ALS/MND), from single motor units to virtual clinical trials.**

`mndsim` models what happens inside a muscle as motor neurons die. It follows every motor unit, the collateral reinnervation that hides the loss, the upper-motor-neuron input that drives and stresses them, and the spread of disease across the body. It turns all of this into the measures clinicians and trialists actually use: a 48-point functional score, FVC, motor unit number estimates (MUNE), CMAP amplitude and survival.

Use it to:

- **Understand.** See why roughly half of a muscle's motor units are gone before a patient notices weakness, and why fast-fatigable units die first.
- **Ask "what if".** Run the same virtual patient under different treatment mechanisms and start times.
- **Design trials.** Simulate thousands of randomised trials to compare endpoints, sample sizes, durations and enrichment strategies, including type-I-error checks.
- **Build digital twins.** Fit a virtual patient to someone's visit history (functional score, FVC, MUNE), then explore counterfactual treatments.

> [!IMPORTANT]
> `mndsim` is a **research and teaching tool**. Parameters are illustrative and calibrated only to population-level summaries from the literature (see [Calibration](#calibration)). It does not predict the course of any real person and is not medical advice.

![Overview of a simulated disease course](docs/images/overview.png)

---

## Install

```bash
pip install "git+https://github.com/ProudStone-576/motor-unit-degeneration-simulator"
# or, to develop:
git clone https://github.com/ProudStone-576/motor-unit-degeneration-simulator
cd motor-unit-degeneration-simulator
pip install -e ".[dev,app]"
```

It needs Python 3.10+ and depends only on NumPy, SciPy, pandas and Matplotlib.

## Quick start

```python
import mndsim
from mndsim.phenotypes import TYPICAL

result = mndsim.simulate(TYPICAL, months=72)  # one virtual patient, 6 years
result.summary()  # symptom onset, survival, slopes, MU loss...
result.plot()  # the figure above
df = result.to_frame()  # tidy table: one row per month

# Same patient, neuroprotection started at diagnosis
onset = result.symptom_onset[0]
treated = mndsim.simulate(TYPICAL, mndsim.TREATMENTS["neuroprotective"].at(onset + 12), months=72)
print(treated.event_time - result.event_time)  # months gained to respiratory failure
```

Or from the command line:

```bash
mndsim list                                          # phenotypes and treatment archetypes
mndsim simulate --phenotype bulbar_onset --treatment neuroprotective --plot course.png
mndsim compare neuroprotective nmj_stabilizer symptomatic --start-after-onset 6 --plot cf.png
mndsim cohort --n 1000 --csv cohort.csv --plot cohort.png
mndsim trial --treatment anti_excitotoxic --sample-sizes 50 100 200 300 --plot power.png
mndsim fit visits.csv --fvc-col fvc --onset-region bulbar --forecast neuroprotective
mndsim sensitivity --plot tornado.png
```

There is also a point-and-click app:

```bash
streamlit run app/streamlit_app.py
```

## What the model captures

Each of four regions (bulbar, respiratory, upper limb, lower limb) has its own pool of 120 motor units.

| Mechanism | How it is modelled | Grounding |
|---|---|---|
| **Size principle and force** | Exponentially distributed twitch forces and recruitment thresholds; rate coding with a sigmoidal force–frequency relation | Henneman 1957; Fuglevand, Winter & Patla 1993 |
| **Selective vulnerability** | Hazard by unit type: fast-fatigable > fast-resistant > slow, plus unit-to-unit variability | Frey et al. 2000; Pun et al. 2006 |
| **Dying-back** | Units pass through a *denervating* state: unreliable transmission and retracting terminals, before the motoneuron is lost | Fischer et al. 2004 |
| **Collateral reinnervation** | Surviving healthy units adopt orphaned fibres, up to a maximum enlargement; FF units sprout poorly | Frey et al. 2000 |
| **Irreversible denervation** | Fibres orphaned for too long degenerate and can no longer be rescued | — |
| **Upper motor neuron loss** | Lowers maximal voluntary drive to each pool | — |
| **Excitotoxicity ("dying forward")** | Glutamatergic drive from surviving UMNs amplifies LMN hazard | Eisen et al. 1992; Vucic et al. 2008 |
| **Focal onset and contiguous spread** | Disease reaches regions with a delay proportional to neuraxis distance | Ravits & La Spada 2009 |
| **Clinical read-outs** | 12-item, 48-point functional score modelled on the ALSFRS-R structure; FVC; MUNE; CMAP; MUP size; respiratory-failure event | Cedarbaum et al. 1999 |

Simulations are **mean-field** (expected trajectories, deterministic) or **stochastic** (individual unit fates, giving the step-wise MUNE decrements seen in serial studies). Both run vectorised over patients × regions × units; a 3,000-patient, 20-year cohort takes about half a minute on a laptop.

The full equations, every parameter, and where each value comes from are in **[docs/MODEL.md](docs/MODEL.md)**.

## Gallery

<table>
<tr>
<td width="50%"><img src="docs/images/motor_units.png" alt="Motor unit raster"><br>
<b>Every motor unit's fate.</b> Large fast-fatigable units are lost first and slow units last. Orange marks the dying-back phase.</td>
<td width="50%"><img src="docs/images/compensation.png" alt="Compensation curve"><br>
<b>The clinical threshold.</b> Reinnervation keeps strength near normal until about half the units are gone. Without it, losing the largest units first makes weakness <i>disproportionate</i>.</td>
</tr>
<tr>
<td><img src="docs/images/treatment_timing.png" alt="Treatment timing"><br>
<b>Timing matters.</b> The same drug started at symptom onset, at diagnosis, or a year later.</td>
<td><img src="docs/images/cohort.png" alt="Virtual cohort"><br>
<b>Virtual cohort.</b> Heterogeneous patients; bulbar onset progresses fastest and lower-limb onset slowest.</td>
</tr>
<tr>
<td><img src="docs/images/power.png" alt="Power curve"><br>
<b>In-silico trial design.</b> Power by endpoint for a modest anti-excitotoxic effect. Survival is badly underpowered in 12 months.</td>
<td><img src="docs/images/sensitivity.png" alt="Sensitivity analysis"><br>
<b>Global sensitivity (LHS + PRCC).</b> Which parameters drive survival after symptom onset.</td>
</tr>
</table>

Regenerate every figure with `python examples/make_figures.py`.

## Tour

### Patients and phenotypes

```python
from mndsim import Patient, PHENOTYPES

list(PHENOTYPES)
# ['typical', 'lower_limb_onset', 'bulbar_onset', 'fast_progressor', 'slow_progressor',
#  'umn_dominant', 'lmn_dominant', 'flail_arm', 'sod1_familial', 'healthy']

me = Patient(
    onset_region="bulbar",
    lmn_rate=0.12,
    spread_rate=0.06,
    region_severity={"lower_limb": 0.5},
    genotype="C9orf72",
)
```

`umn_dominant` (PLS-like) weakens through lost drive with almost no LMN loss. `lmn_dominant` (PMA-like) does the opposite, and `flail_arm` stays confined to the arms for years. Time zero is biological disease onset; **symptom onset is emergent**. It happens when compensation can no longer hide the loss.

### Treatments are mechanisms, not drug names

```python
from mndsim import Treatment, TREATMENTS
# placebo, neuroprotective, anti_excitotoxic, nmj_stabilizer,
# sprouting_enhancer, symptomatic, gene_targeted (SOD1 only)

my_drug = Treatment(
    name="my_compound",
    lmn_protection=0.3,  # -30% LMN hazard
    nmj_rescue=0.05,  # re-stabilise denervating units
    ramp=3.0,  # months to reach full effect
    target_genotype=None,  # or e.g. "SOD1"
)
combo = TREATMENTS["neuroprotective"] + my_drug  # protections compose multiplicatively
mndsim.simulate(me, combo.at(start=24, stop=48))  # with washout after stopping
```

All library effect sizes are **illustrative archetypes**, not estimates for real drugs.

### Cohorts, trials and power

```python
cohort = mndsim.sample_cohort(1000, seed=1)  # heterogeneous virtual population
res = mndsim.simulate(cohort, months=150)
res.summary().groupby("onset_region").survival_from_symptom_onset.median()

design = mndsim.TrialDesign(n_per_arm=150, duration=12, min_fvc=70, min_delta_fs=0.0)
pool = mndsim.build_trial_pool(TREATMENTS["anti_excitotoxic"], design, seed=1)
mndsim.run_trials(pool, n_trials=500).power()  # power per endpoint
mndsim.power_curve(pool, [50, 100, 200, 300])  # sample-size planning
pool.individual_effects()  # per-patient effect: both potential outcomes
```

Each eligible virtual patient is simulated **twice**, untreated and treated from enrolment. Replicate trials then resample the pool, randomise, and add dropout and rater noise. They analyse five endpoints:

- functional slope (Welch's t-test on per-patient slopes)
- survival (log-rank, with a Peto hazard ratio)
- CAFS (Berry et al. 2013)
- MUNE decline in the region of onset
- FVC slope

A placebo pool gives about the nominal false-positive rate. The test suite checks this.

### Digital twins

```python
fit = mndsim.fit_patient(
    months_since_onset=[2, 5, 8, 11, 14],
    alsfrs=[47, 43, 41, 37, 34],
    fvc=[92, 97, 84, 71, 53],
    onset_region="bulbar",
)
fit.confidence_intervals()  # estimates with approximate 95% CIs
fit.counterfactual(TREATMENTS["neuroprotective"], start_after_onset=6)
```

On synthetic data the fit recovers the generating parameters within their confidence intervals (see `tests/` and `examples/04_digital_twin.py`).

### Sensitivity analysis

```python
sa = mndsim.global_sensitivity(n_samples=400)  # Latin hypercube + PRCC
sa.ranking("mu_loss_at_symptom_onset")  # max enlargement dominates (PRCC ~ 0.99)
mndsim.sweep("max_enlargement", [1, 2.5, 5, 8])  # one-at-a-time
```

## Calibration

The defaults were tuned so the virtual population reproduces commonly reported, population-level features of sporadic ALS. They were **not** fitted to individual patient data. Reproduce this table with `python examples/calibration_report.py`.

| Quantity | Model (n = 3,000) | Commonly reported |
|---|---|---|
| Median survival from symptom onset | 32 months | about 2–4 years |
| Alive 5 years after onset | 16% | about 10–20% |
| Median survival by onset: bulbar / upper limb / lower limb | 25 / 29 / 47 months | bulbar onset carries the worst prognosis |
| Functional score slope, first year (median; 10th–90th percentile) | −0.68 (−1.59 to −0.32) points/month | median roughly −0.6 to −1.0, wide and right-skewed |
| Progression rate at diagnosis, ΔFS (median) | 0.83 points/month | strongly prognostic (Kimura et al. 2006) |
| Functional score at diagnosis (median) | 39 / 48 | high 30s in trial populations |
| Motor units lost when weakness appears (median) | 48% | substantial loss precedes weakness |

Treat any single number as one model's output under stated assumptions, not as a fact about ALS.

## Limitations

Read these before drawing conclusions:

- **One pool per region.** Real regions contain many muscles, and laterality (left/right spread) is not modelled.
- **Respiratory failure is the only terminal event.** It stands for "death or permanent ventilation"; bulbar complications, nutrition and NIV are not modelled.
- **Survival is too predictable.** Survival correlates with ΔFS at r ≈ −0.92, far stronger than in real cohorts, because there is no unmodelled clinical variability (care, comorbidity, intercurrent illness).
- **FVC is a proxy.** It equals respiratory pool strength. The lower tail of FVC at diagnosis is probably too heavy; FVC < 70% is the main reason for the roughly 35% trial screen-failure rate.
- **The functional score mimics the structure of the ALSFRS-R, not the instrument.** Item thresholds are illustrative.
- **Treatment effects are mechanism archetypes.** They have no pharmacokinetics beyond first-order ramp and washout.
- **Calibration matches population summaries only.** Individual-level validity (for example, of fitted twins) is untested on real data.
- **No cognition or behaviour.** The frontotemporal spectrum is outside scope.

Issues and pull requests that address these are very welcome.

## Project layout

```
src/mndsim/
  pool.py         motor-unit pool (size principle, force)        engine.py      vectorised simulator
  patient.py      Patient parameters                             result.py      SimulationResult
  phenotypes.py   preset patients                                treatment.py   mechanism-based treatments
  cohort.py       population model                               trial.py       in-silico RCTs, power
  fitting.py      digital-twin fitting                           sensitivity.py LHS + PRCC, sweeps
  scoring.py      functional score, FVC                          plotting.py    figures
  config.py       physiological constants                        cli.py         `mndsim` command
app/              Streamlit explorer
examples/         runnable scripts (quick start, counterfactuals, trial design, digital twin, figures)
docs/MODEL.md     model equations, parameters and references
tests/            pytest suite (invariants, mechanisms, statistics, CLI)
```

## Contributing

Contributions from modellers, clinicians, electrophysiologists and software engineers are all valuable. Better parameter estimates with references are especially welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check .
```

## Citing

If `mndsim` helps your work, please cite it using [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Origins

The project began as a five-node signal-flow prototype (cortex → UMN → LMN → NMJ → muscle) with linear degeneration and a neural-ODE experiment. Version 0.1 rebuilt it around an explicit motor-unit pool, so that compensation, selective vulnerability and electrophysiological read-outs emerge from the mechanism rather than being assumed. The prototype remains in the git history.

## License

[MIT](LICENSE)
