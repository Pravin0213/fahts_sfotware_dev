"""3-D transient conduction in a vessel shell (Hex8 FEM), coupled to the vessel contents."""

from fahts.wall.fem_3d.shell_mesh import PeakZoneGeometry, VesselShellMesh
from fahts.wall.fem_3d.shell_solver import ShellConduction3D

__all__ = ["PeakZoneGeometry", "ShellConduction3D", "VesselShellMesh"]
