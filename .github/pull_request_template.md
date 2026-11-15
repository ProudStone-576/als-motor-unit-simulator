## What does this change?

## Why?

## How was it checked?
- [ ] `pytest` passes
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] If model behaviour changed: calibration targets in `tests/test_engine.py` still hold, figures regenerated with `python examples/make_figures.py`, and `docs/MODEL.md` updated
