"""Global physiological constants of the model.

Patient-specific quantities (rates of degeneration, spread, sprouting...)
live on :class:`mndsim.Patient`. Everything here is shared by all patients
and describes the motor system itself. Defaults are documented in
``docs/MODEL.md`` together with their provenance; values marked
*illustrative* are plausible choices, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .regions import FUNCTIONAL_ITEMS, FunctionalItem

#: Motor-unit types in size order: slow (S), fast fatigue-resistant (FR),
#: fast fatigable (FF).
UNIT_TYPES: tuple[str, ...] = ("S", "FR", "FF")


@dataclass(frozen=True)
class PhysiologyConfig:
    """Constants describing the healthy motor system and disease mechanics.

    Pool structure follows Fuglevand, Winter & Patla (1993), *J Neurophysiol*
    70:2470-2488: exponentially distributed twitch forces and recruitment
    thresholds (Henneman's size principle), linear rate coding with a
    sigmoidal force-frequency relation.
    """

    # --- Motor-unit pool (Fuglevand et al., 1993) -------------------------
    n_units: int = 120
    """Motor units per regional pool (120 as in Fuglevand et al., 1993)."""
    twitch_force_range: float = 100.0
    """Ratio of largest to smallest twitch force (RP)."""
    recruitment_range: float = 30.0
    """Ratio of highest to lowest recruitment threshold (RR)."""
    min_firing_rate: float = 8.0
    """Firing rate at recruitment, Hz (MFR)."""
    peak_firing_rate_first: float = 35.0
    """Peak firing rate of the first-recruited unit, Hz (PFR1)."""
    peak_firing_rate_diff: float = 10.0
    """Peak-rate difference between first and last recruited unit, Hz (PFRD)."""
    excitation_gain: float = 1.0
    """Firing-rate gain, Hz per unit excitation (g_e)."""
    longest_contraction_time: float = 0.090
    """Twitch contraction time of the smallest unit, s (T_L)."""
    contraction_time_range: float = 3.0
    """Ratio of longest to shortest contraction time (RT)."""
    type_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2)
    """Fraction of units (by size rank) that are S, FR and FF (illustrative)."""

    # --- Selective vulnerability & plasticity ----------------------------
    type_vulnerability: tuple[float, float, float] = (0.35, 1.0, 2.2)
    """Relative degeneration hazard of S, FR and FF units. FF units are lost
    first in ALS (Frey et al., 2000; Pun et al., 2006); magnitudes are
    illustrative."""
    unit_vulnerability_sd: float = 0.35
    """Log-normal SD of unit-to-unit vulnerability within a type."""
    type_sprouting: tuple[float, float, float] = (1.0, 0.7, 0.3)
    """Relative collateral-sprouting competence of S, FR and FF units
    (FF units sprout poorly; Frey et al., 2000). Illustrative."""

    # --- Dying-back and denervation (months) -------------------------------
    dying_back_duration: float = 4.0
    """Mean time a unit spends denervating (terminals retracting) before the
    motoneuron soma dies. Captures 'dying-back' (Fischer et al., 2004)."""
    terminal_retraction_time: float = 3.0
    """Time constant of terminal retraction in a denervating unit."""
    denervating_transmission: float = 0.6
    """Neuromuscular transmission reliability of a denervating unit
    (failing/jittery junctions)."""
    denervated_fiber_survival: float = 15.0
    """Time constant after which orphaned muscle fibres that were never
    reinnervated degenerate irreversibly (illustrative)."""

    # --- Upper motor neuron -------------------------------------------------
    residual_drive: float = 0.5
    """Fraction of maximal voluntary excitation that survives complete
    corticospinal loss (non-pyramidal descending pathways). Illustrative."""

    # --- Spread -------------------------------------------------------------
    regional_ramp_time: float = 2.0
    """Time constant over which disease activity ramps up once it reaches
    a region (months)."""

    # --- Neurofilament light chain (NfL) ----------------------------------
    nfl_healthy: float = 10.0
    """Plasma NfL of a healthy 60-year-old, pg/mL. The model's NfL scale is
    illustrative: use ratios to healthy rather than absolute values."""
    nfl_age_slope: float = 0.022
    """Log-linear rise of healthy NfL with age, per year (NfL increases with
    normal ageing)."""
    nfl_half_life: float = 1.0
    """Plasma NfL elimination half-life, months (illustrative)."""
    nfl_release: float = 3.3
    """pg/mL added per month by degeneration of one axon-equivalent per month
    (scale factor; sets the ALS-to-healthy ratio)."""
    nfl_distal_fraction: float = 0.3
    """Fraction of a lower motor neuron's NfL released when it starts dying
    back (distal axon); the rest is released when the neuron is lost."""
    nfl_size_exponent: float = 0.25
    """Axonal NfL content scales as (unit size)^exponent, normalised to mean 1:
    large motor units have thicker axons."""
    nfl_umn_equivalents: float = 150.0
    """NfL content of a region's whole upper-motor-neuron population, in
    lower-motor-unit axon equivalents (illustrative)."""
    axon_length: tuple[float, float, float, float] = (0.25, 0.5, 0.8, 1.0)
    """Relative axon length of bulbar, respiratory, upper- and lower-limb
    pathways; longer axons contain more neurofilament."""

    # --- Clinical read-outs -------------------------------------------------
    symptom_threshold: float = 0.80
    """Regional strength (fraction of healthy MVC) below which weakness is
    first noticed; defines symptom onset."""
    respiratory_failure_threshold: float = 0.30
    """Respiratory capacity (FVC proxy, fraction of predicted) defining the
    composite 'death or permanent assisted ventilation' event."""
    functional_items: tuple[FunctionalItem, ...] = field(default=FUNCTIONAL_ITEMS)

    def __post_init__(self) -> None:
        if self.n_units < 3:
            raise ValueError("n_units must be >= 3")
        if abs(sum(self.type_fractions) - 1.0) > 1e-9:
            raise ValueError("type_fractions must sum to 1")
        for name in ("twitch_force_range", "recruitment_range", "contraction_time_range"):
            if getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be > 1")
        for name in (
            "dying_back_duration",
            "terminal_retraction_time",
            "denervated_fiber_survival",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not all(0.0 <= k <= 1.0 for k in self.type_sprouting):
            raise ValueError("type_sprouting values must be in [0, 1]")
        if any(v < 0 for v in self.type_vulnerability):
            raise ValueError("type_vulnerability values must be >= 0")
        if self.nfl_half_life <= 0 or self.nfl_healthy <= 0 or self.nfl_release < 0:
            raise ValueError("NfL half-life and healthy level must be > 0, release >= 0")
        if not 0.0 <= self.nfl_distal_fraction <= 1.0:
            raise ValueError("nfl_distal_fraction must be in [0, 1]")
        if len(self.axon_length) != 4 or any(x < 0 for x in self.axon_length):
            raise ValueError("axon_length needs four non-negative values")
        if not 0.0 <= self.residual_drive <= 1.0:
            raise ValueError("residual_drive must be in [0, 1]")

    def replace(self, **changes) -> PhysiologyConfig:
        """Return a copy with some fields changed."""
        return replace(self, **changes)


DEFAULT_CONFIG = PhysiologyConfig()
