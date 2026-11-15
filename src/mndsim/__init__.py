"""mndsim: a mechanistic motor-unit degeneration simulator for ALS/MND research.

Quick start::

    import mndsim
    from mndsim.phenotypes import TYPICAL

    result = mndsim.simulate(TYPICAL, months=72)
    print(result.summary())
    result.plot()

See the README and ``docs/MODEL.md`` for the model description.
"""

from . import hidden_phase, phenotypes
from .cohort import DEFAULT_POPULATION, PopulationModel, sample_cohort
from .config import DEFAULT_CONFIG, UNIT_TYPES, PhysiologyConfig
from .engine import simulate
from .fitting import FitResult, fit_patient
from .patient import Patient
from .phenotypes import PHENOTYPES, get_phenotype
from .pool import MotorUnitPool
from .regions import FUNCTIONAL_ITEMS, REGIONS
from .result import SimulationResult
from .sensitivity import global_sensitivity, prcc, sweep
from .treatment import TREATMENTS, Combination, Treatment, get_treatment
from .trial import (
    TrialDesign,
    TrialPool,
    TrialResults,
    build_trial_pool,
    power_by_duration,
    power_curve,
    run_trials,
    simulate_trial,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_POPULATION",
    "FUNCTIONAL_ITEMS",
    "PHENOTYPES",
    "REGIONS",
    "TREATMENTS",
    "UNIT_TYPES",
    "Combination",
    "FitResult",
    "MotorUnitPool",
    "Patient",
    "PhysiologyConfig",
    "PopulationModel",
    "SimulationResult",
    "Treatment",
    "TrialDesign",
    "TrialPool",
    "TrialResults",
    "__version__",
    "build_trial_pool",
    "fit_patient",
    "get_phenotype",
    "get_treatment",
    "global_sensitivity",
    "hidden_phase",
    "phenotypes",
    "power_by_duration",
    "power_curve",
    "prcc",
    "run_trials",
    "sample_cohort",
    "simulate",
    "simulate_trial",
    "sweep",
]
