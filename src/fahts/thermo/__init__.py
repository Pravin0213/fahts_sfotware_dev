"""Thermodynamics of the vessel contents.

Peng-Robinson mixture model with flash algorithms (PT, UV, PH, PS), free water,
pseudo-components, saturation and transport properties. Pure physics: no knowledge of
vessels, walls or fire.

    from fahts.thermo import PRMixture
    m = PRMixture({"C1": 0.9, "C3": 0.1})
    r = m.flash_PT(50e5, 293.15)
"""

from fahts.thermo.pr_mixture import PRMixture
from fahts.thermo.pseudo import characterise_pseudo
from fahts.thermo.results import FlashResult, Phase

__all__ = ["PRMixture", "FlashResult", "Phase", "characterise_pseudo"]
