"""
BOX cross-section 3-D surface mesh generator (FAHTS approach).

Creates Quad4 shell elements on the four outer faces of a hollow rectangular
(BOX / RHS) beam.  The 2-D heat equation is solved in the axial × hoop plane
of each element; wall thickness is a scalar parameter, not a mesh dimension
(SINTEF FAHTS §3.2.2, §3.4.1).

Default mesh divisions (matching FAHTS software defaults):
    n_top    = 2   elements across top / bottom face (width / hoop direction)
    n_side   = 3   elements across left / right face (height / hoop direction)
    n_length = 4   elements along beam axis

Face layout (beam-local coordinates):
    Bottom face  z = −H/2,  y ∈ [−W/2, +W/2],  thickness = T_bot
    Top    face  z = +H/2,  y ∈ [−W/2, +W/2],  thickness = T_top
    Left   face  y = −W/2,  z ∈ [−H/2, +H/2],  thickness = T_side
    Right  face  y = +W/2,  z ∈ [−H/2, +H/2],  thickness = T_side

Corner nodes (e.g. x_i, ±W/2, ±H/2) are shared between adjacent faces via
3-D coordinate deduplication with 1e-10 m tolerance.
"""
from __future__ import annotations
import numpy as np

from fahts.core.model.section import BoxSection
from .beam_surface_mesh import BeamSurfaceMesh

_DEDUP_TOL = 1e-10


class BoxSurfaceMesher:
    """
    Generates the FAHTS-style surface Quad4 mesh for a hollow BOX beam.

    Args:
        section:  BoxSection to mesh.
        length:   Beam element length [m].
        n_top:    Elements across top / bottom face (default 2).
        n_side:   Elements across left / right face (default 3).
        n_length: Elements along beam axis (default 4).
    """

    _DEFAULT_N_TOP    = 2
    _DEFAULT_N_SIDE   = 3
    _DEFAULT_N_LENGTH = 4

    def __init__(
        self,
        section: BoxSection,
        length: float,
        n_top: int = _DEFAULT_N_TOP,
        n_side: int = _DEFAULT_N_SIDE,
        n_length: int = _DEFAULT_N_LENGTH,
    ) -> None:
        if n_top < 1 or n_side < 1 or n_length < 1:
            raise ValueError(
                f"n_top ({n_top}), n_side ({n_side}), n_length ({n_length}) "
                "must all be ≥ 1"
            )
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self._sec      = section
        self._L        = length
        self._n_top    = n_top
        self._n_side   = n_side
        self._n_length = n_length

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> BeamSurfaceMesh:
        """Build and return the BeamSurfaceMesh for the BOX beam."""
        s = self._sec
        L = self._L

        hy = s.W / 2.0   # half-width  (y-direction)
        hz = s.H / 2.0   # half-height (z-direction)

        xs = np.linspace(0.0,  L,  self._n_length + 1)
        ys = np.linspace(-hy, hy, self._n_top    + 1)
        zs = np.linspace(-hz, hz, self._n_side   + 1)

        # Global node pool with 3-D coordinate deduplication
        pool: list[np.ndarray] = []
        idx:  dict[tuple, int] = {}

        def gid(x: float, y: float, z: float) -> int:
            key = (_snap(x), _snap(y), _snap(z))
            if key not in idx:
                idx[key] = len(pool)
                pool.append(np.array([x, y, z], dtype=float))
            return idx[key]

        quads:       list[tuple[int, int, int, int]] = []
        thicknesses: list[float]                     = []
        local_cols:  list[tuple[int, int]]           = []

        def add_face_xy(z_const: float, thickness: float) -> None:
            """Add n_length × n_top quads on a constant-z (top/bottom) face."""
            for ix in range(self._n_length):
                for iy in range(self._n_top):
                    n0 = gid(xs[ix],     ys[iy],     z_const)
                    n1 = gid(xs[ix + 1], ys[iy],     z_const)
                    n2 = gid(xs[ix + 1], ys[iy + 1], z_const)
                    n3 = gid(xs[ix],     ys[iy + 1], z_const)
                    quads.append((n0, n1, n2, n3))
                    thicknesses.append(thickness)
                    local_cols.append((0, 1))   # FEM in (x, y) plane

        def add_face_xz(y_const: float, thickness: float) -> None:
            """Add n_length × n_side quads on a constant-y (left/right) face."""
            for ix in range(self._n_length):
                for iz in range(self._n_side):
                    n0 = gid(xs[ix],     y_const, zs[iz])
                    n1 = gid(xs[ix + 1], y_const, zs[iz])
                    n2 = gid(xs[ix + 1], y_const, zs[iz + 1])
                    n3 = gid(xs[ix],     y_const, zs[iz + 1])
                    quads.append((n0, n1, n2, n3))
                    thicknesses.append(thickness)
                    local_cols.append((0, 2))   # FEM in (x, z) plane

        add_face_xy(-hz, s.T_bot)   # bottom
        add_face_xy(+hz, s.T_top)   # top
        add_face_xz(-hy, s.T_side)  # left
        add_face_xz(+hy, s.T_side)  # right

        return BeamSurfaceMesh(
            nodes=np.array(pool, dtype=float),
            quads=np.array(quads, dtype=np.intp),
            thicknesses=np.array(thicknesses, dtype=float),
            local_cols=np.array(local_cols, dtype=np.intp),
        )


# ── Module helpers ────────────────────────────────────────────────────────────

def _snap(v: float) -> float:
    return round(v / _DEDUP_TOL) * _DEDUP_TOL
