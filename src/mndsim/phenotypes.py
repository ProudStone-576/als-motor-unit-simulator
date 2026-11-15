"""Preset virtual patients spanning the clinical spectrum of MND.

Parameters were tuned so the *typical* phenotype reproduces commonly
reported population-level summaries of sporadic ALS (survival from
symptom onset of roughly 2.5-3 years, functional decline of roughly
0.8-1 point per month on a 48-point scale, and loss of about half of the
motor units of the affected region before weakness is noticed). Other
phenotypes are qualitative variations on that baseline. None is fitted to
individual patient data.
"""

from __future__ import annotations

from .patient import Patient

TYPICAL = Patient(name="typical", onset_region="upper_limb")
"""Sporadic, upper-limb-onset ALS with mixed UMN and LMN involvement."""

LOWER_LIMB_ONSET = Patient(name="lower_limb_onset", onset_region="lower_limb")

BULBAR_ONSET = Patient(
    name="bulbar_onset",
    onset_region="bulbar",
    lmn_rate=0.10,
    spread_rate=0.05,
    age_at_onset=66.0,
)

FAST_PROGRESSOR = Patient(
    name="fast_progressor",
    lmn_rate=0.16,
    spread_rate=0.07,
    excitotoxicity=0.8,
)

SLOW_PROGRESSOR = Patient(
    name="slow_progressor",
    lmn_rate=0.035,
    spread_rate=0.02,
    excitotoxicity=0.3,
)

UMN_DOMINANT = Patient(
    name="umn_dominant",
    umn_rate=0.05,
    lmn_rate=0.008,
    excitotoxicity=0.2,
)
"""Primary-lateral-sclerosis-like: weakness driven by upper motor neuron loss."""

LMN_DOMINANT = Patient(
    name="lmn_dominant",
    umn_rate=0.002,
    lmn_rate=0.085,
    excitotoxicity=0.2,
)
"""Progressive-muscular-atrophy-like: lower motor neuron disease with spared UMNs."""

FLAIL_ARM = Patient(
    name="flail_arm",
    onset_region="upper_limb",
    spread_rate=0.012,
)
"""Brachial amyotrophic diplegia: long confinement to the upper limbs."""

SOD1_FAMILIAL = Patient(
    name="sod1_familial",
    onset_region="lower_limb",
    lmn_rate=0.14,
    umn_rate=0.003,
    spread_rate=0.06,
    genotype="SOD1",
    age_at_onset=48.0,
)
"""LMN-predominant, lower-limb-onset familial disease (target of gene therapy)."""

HEALTHY = Patient(name="healthy", lmn_rate=0.0, umn_rate=0.0, excitotoxicity=0.0)
"""No degeneration; useful as a control."""

PHENOTYPES: dict[str, Patient] = {
    p.name: p
    for p in (
        TYPICAL,
        LOWER_LIMB_ONSET,
        BULBAR_ONSET,
        FAST_PROGRESSOR,
        SLOW_PROGRESSOR,
        UMN_DOMINANT,
        LMN_DOMINANT,
        FLAIL_ARM,
        SOD1_FAMILIAL,
        HEALTHY,
    )
}


def get_phenotype(name: str) -> Patient:
    """Look up a preset by name."""
    try:
        return PHENOTYPES[name]
    except KeyError:
        raise KeyError(f"unknown phenotype {name!r}; available: {sorted(PHENOTYPES)}") from None
