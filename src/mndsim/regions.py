"""Body regions and the synthetic functional rating scale.

The simulator follows four functional regions, each with its own
motor-unit pool:

* ``bulbar``      - brainstem motor nuclei (speech, swallowing)
* ``respiratory`` - phrenic (C3-C5) and thoracic intercostal pools
* ``upper_limb``  - brachial plexus pools (C5-T1)
* ``lower_limb``  - lumbosacral pools (L1-S2)

Disease starts focally in one region and spreads contiguously along the
neuraxis (Ravits & La Spada, 2009). Spread delay is proportional to the
distance between *effective* neuraxis positions.
"""

from __future__ import annotations

from dataclasses import dataclass

REGIONS: tuple[str, ...] = ("bulbar", "respiratory", "upper_limb", "lower_limb")
"""Region names, in the order used along the ``region`` axis of all arrays."""

#: Effective rostro-caudal position of each region along the neuraxis
#: (arbitrary units; 1 unit is roughly the brainstem-to-brachial distance).
#: "respiratory" lies between the bulbar and brachial pools because its
#: function depends on both phrenic (C3-C5) and intercostal (thoracic) pools.
NEURAXIS_POSITION: dict[str, float] = {
    "bulbar": 0.0,
    "respiratory": 0.5,
    "upper_limb": 1.0,
    "lower_limb": 1.7,
}


def region_index(name: str) -> int:
    """Return the array index of a region, with a helpful error message."""
    try:
        return REGIONS.index(name)
    except ValueError:
        raise ValueError(f"unknown region {name!r}; expected one of {REGIONS}") from None


@dataclass(frozen=True)
class FunctionalItem:
    """One item of the synthetic functional rating scale (scored 0-4).

    The item score is a smooth, monotone function of the strength
    ("capacity", 1 = healthy) of the region(s) that drive it. ``c50`` is
    the capacity at which the item is half-lost; items with higher ``c50``
    are lost earlier (e.g. climbing stairs before turning in bed).
    """

    name: str
    domain: str
    regions: tuple[str, ...]
    c50: float
    width: float = 0.15


#: Twelve items modelled on the *structure* of the ALSFRS-R (Cedarbaum et
#: al., 1999): four domains of three items, each scored 0-4, total 0-48.
#: This is a synthetic, mechanistically-driven analogue, not the clinical
#: instrument; item thresholds are illustrative.
FUNCTIONAL_ITEMS: tuple[FunctionalItem, ...] = (
    FunctionalItem("speech", "bulbar", ("bulbar",), 0.55),
    FunctionalItem("salivation", "bulbar", ("bulbar",), 0.40),
    FunctionalItem("swallowing", "bulbar", ("bulbar",), 0.45),
    FunctionalItem("handwriting", "fine_motor", ("upper_limb",), 0.60),
    FunctionalItem("cutting_food", "fine_motor", ("upper_limb",), 0.50),
    FunctionalItem("dressing_hygiene", "fine_motor", ("upper_limb",), 0.45),
    FunctionalItem("turning_in_bed", "gross_motor", ("upper_limb", "lower_limb"), 0.40),
    FunctionalItem("walking", "gross_motor", ("lower_limb",), 0.50),
    FunctionalItem("climbing_stairs", "gross_motor", ("lower_limb",), 0.62),
    FunctionalItem("dyspnea", "respiratory", ("respiratory",), 0.60),
    FunctionalItem("orthopnea", "respiratory", ("respiratory",), 0.50),
    FunctionalItem("respiratory_insufficiency", "respiratory", ("respiratory",), 0.35),
)

ITEM_NAMES: tuple[str, ...] = tuple(item.name for item in FUNCTIONAL_ITEMS)
DOMAINS: tuple[str, ...] = ("bulbar", "fine_motor", "gross_motor", "respiratory")
MAX_FUNCTIONAL_SCORE: int = 4 * len(FUNCTIONAL_ITEMS)
