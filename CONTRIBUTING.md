# Contributing to mndsim

Thank you for helping. This project sits between neuroscience, clinical research and software, so contributions of very different kinds are valuable:

- **Better science.** A parameter value with a reference, a mechanism that is missing or wrong, a calibration target. Open a "Model / science question" issue with the citation (DOI or PubMed ID).
- **Validation.** Comparisons against published natural-history or trial data, as notebooks or scripts in `examples/`.
- **Features.** New endpoints or trial designs, regions or muscles, treatment mechanisms, faster engines.
- **Docs and teaching material.**

## Development setup

```bash
git clone https://github.com/ProudStone-576/motor-unit-degeneration-simulator
cd motor-unit-degeneration-simulator
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
pytest                       # about 45 s
ruff check . && ruff format --check .
python web/tests/make_golden.py && node --test web/tests/parity.test.mjs   # browser engine parity
node web/build_site.mjs && python -m http.server -d _site 8000           # preview the web app
```

## Ground rules

1. **Never commit environments, archives or caches.** `.gitignore` covers the common cases.
2. **Model changes need evidence.** If behaviour changes:
   - say why, with references where possible;
   - keep the calibration tests in `tests/test_engine.py` passing, or update them deliberately and explain why;
   - re-run `python examples/calibration_report.py` and update the README table;
   - regenerate figures with `python examples/make_figures.py`;
   - update `docs/MODEL.md`.
3. **Test invariants, not just numbers.** Good tests check properties that must hold whatever the parameter values: fibre conservation, monotonicity, placebo equals untreated, mean-field equals the stochastic average, type-I error control.
4. **Keep the two engines in sync.** The browser app runs `web/mndsim.js`, a port of the Python engine. If you change the model, change both, then run `python web/tests/make_golden.py && node --test web/tests/parity.test.mjs`. CI runs this parity check on every push.
5. **Keep the engine vectorised.** No per-patient Python loops in the hot path. Check speed with a 1,000-patient cohort before and after.
6. **No real patient data in the repository.** Example data must be synthetic and labelled as such.
7. **Public functions need docstrings** that state units (months, % predicted, fraction of baseline).

## Pull requests

Keep them focused, describe *what* and *why*, and fill in the checklist in the template. CI runs lint and tests on Linux (Python 3.10–3.13), macOS and Windows, plus Python 3.10 with the oldest supported dependencies.
