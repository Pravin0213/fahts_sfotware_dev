"""Coupled models: the only package that combines wall, fire, contents and relief.

- ``vessel_fire_1d``: vessel in fire, two-zone contents + 1-D wall columns
"""

from fahts.coupling.options import VesselFireOptions
from fahts.coupling.vessel_fire_1d import VesselFireModel, simulate

__all__ = ["VesselFireModel", "VesselFireOptions", "simulate"]
