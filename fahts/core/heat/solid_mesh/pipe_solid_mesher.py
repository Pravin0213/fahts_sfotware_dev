"""
PIPE (circular hollow) Hex8 solid mesher.

Beam-local x = axis ∈ [0, L]; the annulus is centred on the origin in y-z.
Nodes lie exactly on n_layers + 1 concentric circles from R_o = D/2 down to
R_i = R_o − t, at angles θ_j = 2πj / c_circ (θ measured from +y towards +z,
same as PipeSurfaceMesher).  The periodic seam (j = c_circ ≡ 0) shares nodes.
The meshed volume is that of the inscribed polygonal annulus:
    V = ½ c sin(2π/c) (R_o² − R_i²) L  →  π (R_o² − R_i²) L  as c → ∞.

Faces: outer circle → FACE_OUTER, bore → FACE_INNER, x = 0 / L → FACE_END.
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import PipeSection
from ._hex_topology import ring_section
from .extrude import extrude_section
from .solid_mesh import SolidMesh


class PipeSolidMesher:
    """
    Hex8 solid mesh of a PIPE beam.

    Args:
        section:  PipeSection to mesh.
        length:   Member length [m].
        c_circ:   Hoop elements around the circumference (default 16, ≥ 3).
        n_length: Axial elements (default 4).
        n_layers: Radial elements through the wall (default 2).
    """

    def __init__(
        self,
        section: PipeSection,
        length: float,
        c_circ: int = 16,
        n_length: int = 4,
        n_layers: int = 2,
    ) -> None:
        if c_circ < 3:
            raise ValueError(f"c_circ must be >= 3, got {c_circ}")
        if n_length < 1 or n_layers < 1:
            raise ValueError(f"n_length ({n_length}), n_layers ({n_layers}) must be >= 1")
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        r_o = section.outer_diameter / 2.0
        if not 0.0 < section.thickness < r_o:
            raise ValueError(
                f"PIPE section {section.sid}: thickness must be in (0, D/2), "
                f"got {section.thickness}"
            )
        self._sec = section
        self._L = length
        self._c = c_circ
        self._n_length = n_length
        self._n_layers = n_layers

    def build(self) -> SolidMesh:
        """Build and return the validated SolidMesh."""
        r_o = self._sec.outer_diameter / 2.0
        r_i = r_o - self._sec.thickness
        radii = r_o + (r_i - r_o) * np.arange(self._n_layers + 1) / self._n_layers
        radii[-1] = r_i
        th = 2.0 * np.pi * np.arange(self._c) / self._c
        loops = np.stack([
            radii[:, None] * np.cos(th)[None, :],
            radii[:, None] * np.sin(th)[None, :],
        ], axis=2)                                                  # (n_l+1, c, 2)
        sec2d = ring_section(loops)
        return extrude_section(sec2d, self._L, self._n_length, section_kind="PIPE")
