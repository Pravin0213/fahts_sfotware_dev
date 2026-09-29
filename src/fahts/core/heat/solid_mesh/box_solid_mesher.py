"""
BOX (hollow rectangular) Hex8 solid mesher.

Geometry follows BoxSurfaceMesher (NOT the BoxSection docstring): beam-local
x = axis ∈ [0, L], **W along y** (outer faces y = ±W/2), **H along z** (outer
faces z = ±H/2), section centred on the origin in y-z.  Walls: top plate T_top
(z > 0), bottom plate T_bot, side walls T_side.

Cross-section: an ORTHOGONAL tensor-product ring.  The y-lines are
    [−W/2, −W/2+T_side] (n_layers) | inner width (n_top) | [W/2−T_side, W/2] (n_layers)
and the z-lines [−H/2, −H/2+T_bot] (n_layers) | inner height (n_side) | top wall
(n_layers); the cavity cells are removed.  Every cell is a rectangle (corner blocks are
T_side × T_bot / T_top rectangles shared by both walls), so all hexes are rectangular
bricks — required for the monotone two-point conduction stencil to be consistent (the
earlier diagonal-trapezoid corners were skewed).  Nodes conform at the corners and the
steel area is exact.
Faces: outer rectangle → FACE_OUTER, cavity → FACE_INNER, x = 0 / L → FACE_END.
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import BoxSection
from fahts.core.heat.section_mesh.section_mesh import SectionMesh

from ._hex_topology import boundary_edges_2d
from .extrude import extrude_section
from .solid_mesh import SolidMesh


class BoxSolidMesher:
    """
    Hex8 solid mesh of a hollow BOX beam.

    Args:
        section:  BoxSection to mesh.
        length:   Member length [m].
        n_top:    Hoop elements across the top / bottom walls (default 2).
        n_side:   Hoop elements along each side wall (default 3).
        n_length: Axial elements (default 4).
        n_layers: Elements through each wall thickness (default 2).
    """

    def __init__(
        self,
        section: BoxSection,
        length: float,
        n_top: int = 2,
        n_side: int = 3,
        n_length: int = 4,
        n_layers: int = 2,
    ) -> None:
        if min(n_top, n_side, n_length, n_layers) < 1:
            raise ValueError(
                f"n_top ({n_top}), n_side ({n_side}), n_length ({n_length}), "
                f"n_layers ({n_layers}) must all be >= 1"
            )
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        if section.inner_width <= 0.0 or section.inner_height <= 0.0:
            raise ValueError(f"BOX section {section.sid} has no hollow interior")
        self._sec = section
        self._L = length
        self._n_top = n_top
        self._n_side = n_side
        self._n_length = n_length
        self._n_layers = n_layers

    def build(self) -> SolidMesh:
        """Build and return the validated SolidMesh."""
        s, nl = self._sec, self._n_layers
        ys = np.concatenate([
            np.linspace(-s.W / 2, -s.W / 2 + s.T_side, nl + 1),
            np.linspace(-s.W / 2 + s.T_side, s.W / 2 - s.T_side, self._n_top + 1)[1:],
            np.linspace(s.W / 2 - s.T_side, s.W / 2, nl + 1)[1:],
        ])
        zs = np.concatenate([
            np.linspace(-s.H / 2, -s.H / 2 + s.T_bot, nl + 1),
            np.linspace(-s.H / 2 + s.T_bot, s.H / 2 - s.T_top, self._n_side + 1)[1:],
            np.linspace(s.H / 2 - s.T_top, s.H / 2, nl + 1)[1:],
        ])
        ny, nz = len(ys), len(zs)
        cells = [(i, j) for j in range(nz - 1) for i in range(ny - 1)
                 if not (nl <= i < nl + self._n_top and nl <= j < nl + self._n_side)]
        used = sorted({(i + di, j + dj) for i, j in cells for di in (0, 1) for dj in (0, 1)})
        nid = {ij: k for k, ij in enumerate(used)}
        nodes = np.array([[ys[i], zs[j]] for i, j in used], dtype=float)
        quads = np.array([[nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]]
                          for i, j in cells], dtype=np.intp)
        edges = boundary_edges_2d(quads)
        p = nodes[edges]                                           # (n_e, 2, 2)
        tol = 1e-9 * max(s.W, s.H)
        on_outer = (
            (np.abs(np.abs(p[:, :, 0]) - s.W / 2) < tol).all(axis=1)
            | (np.abs(p[:, :, 1] + s.H / 2) < tol).all(axis=1)
            | (np.abs(p[:, :, 1] - s.H / 2) < tol).all(axis=1)
        )
        sec2d = SectionMesh(nodes=nodes, quads=quads, outer_edge_pairs=edges[on_outer],
                            inner_edge_pairs=edges[~on_outer])
        return extrude_section(sec2d, self._L, self._n_length, section_kind="BOX")
