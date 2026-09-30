"""Coupled models: the only package that combines wall, fire, contents and relief.

- ``vessel_case``: ``VesselCase`` - inputs of one vessel-in-fire analysis (JSON case files)
- ``runner``: ``run_case`` - run a case (model + rupture evaluation), progress / cancel
- ``vessel_fire_1d``: ``VesselFireModel`` - two-zone contents + 1-D wall regions
- ``vessfire_import``: import VessFire input decks as a ``VesselCase``
"""

from fahts.coupling.options import VesselFireOptions
from fahts.coupling.runner import CaseResult, run_case
from fahts.coupling.vessel_case import VesselCase
from fahts.coupling.vessel_fire_1d import SimulationCancelled, VesselFireModel, simulate

__all__ = [
    "CaseResult",
    "SimulationCancelled",
    "VesselCase",
    "VesselFireModel",
    "VesselFireOptions",
    "run_case",
    "simulate",
]
