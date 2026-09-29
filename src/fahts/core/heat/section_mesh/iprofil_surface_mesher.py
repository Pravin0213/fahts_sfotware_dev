"""
I-profile (IHPROFIL) 3-D surface mesh generator (FAHTS axial × hoop approach).

Creates Quad4 shell elements on the three structural plates of an I/H-beam
following SINTEF FAHTS §3.4.1:

    - Top flange    : inner (bottom) face at z = z_top_in, full width bf_top
    - Web           : left face at y = −tw/2, spanning z_bot_in → z_top_in
    - Bottom flange : inner (top) face at z = z_bot_in, full width bf_bot

The flanges are meshed at their inner faces and the y-grid is forced to include
y = −tw/2 (left web face position).  This makes the web top/bottom corner nodes
(x_i, −tw/2, z_top_in) and (x_i, −tw/2, z_bot_in) coincide exactly with inner-
flange nodes → gid() deduplication creates SHARED nodes, giving a connected FEM
conductivity matrix with heat-conduction paths from web to flanges and back.

The web is placed at y = −tw/2 (not y = 0) so its outward normal (0, −1, 0) is
non-zero and correct for directional (RadiationBall) flux calculations.

Per FAHTS §3.4.1, I/H profiles are OPEN sections with "2 outsides": both faces
of each structural plate are fire-exposed.  This is handled by passing
n_exposed_sides=2 to SurfaceTransientSolver (see analysis_runner.py).  Using ONE
face per plate (rather than separate top+underside faces) avoids double-counting
the plate thermal mass.

Default mesh divisions (matching FAHTS software defaults):
    n_top    = 4   elements across top flange width (hoop direction)
    n_side   = 2   elements along web height (hoop direction)
    n_bottom = 2   elements across bottom flange width (hoop direction)
    n_length = 2   elements along beam axis

Face layout (beam-local: x = axial, y = hoop-horizontal, z = vertical):

  z = z_top_in  ┌─── top flange inner face (n_top × n_length) ──────┐
                │  t = tf_top; y=−tw/2 node ← web junction node     │
  z = z_top_in  └──── web top edge (y=−tw/2) ────────────────────────┘
                              │  web (n_side × n_length)              │
                              │  left face at y = −tw/2               │
  z = z_bot_in  ┌──── web bot edge (y=−tw/2) ────────────────────────┐
                │  t = tf_bot; y=−tw/2 node ← web junction node     │
  z = z_bot_in  └─── bot flange inner face (n_bot × n_length) ───────┘
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import ISection
from .beam_surface_mesh import BeamSurfaceMesh

_DEDUP_TOL = 1e-10


class IProfileSurfaceMesher:
    """
    Generates the FAHTS-style surface Quad4 mesh for an I/H-profile beam.

    Three structural plates are meshed (top flange, web, bottom flange).  Both
    faces of each plate are fire-exposed (§3.4.1 "2 outsides"), so the caller
    must set n_exposed_sides=2 in SurfaceTransientSolver.

    The mesh is topologically connected: inner flange faces (z = z_top_in /
    z_bot_in) include y = −tw/2 as a forced grid node that coincides with the
    web left-face corner.  gid() deduplication creates shared nodes at the
    T-junctions so the FEM conductivity matrix couples web and flange DOFs.

    Args:
        section:   ISection to mesh.
        length:    Beam element length [m].
        n_top:     Elements across top flange width (default 4).
        n_side:    Elements along web height (default 2).
        n_bottom:  Elements across bottom flange width (default 2).
        n_length:  Elements along beam axis (default 2).
    """

    _DEFAULT_N_TOP    = 4
    _DEFAULT_N_SIDE   = 2
    _DEFAULT_N_BOTTOM = 2
    _DEFAULT_N_LENGTH = 2

    def __init__(
        self,
        section: ISection,
        length: float,
        n_top: int = _DEFAULT_N_TOP,
        n_side: int = _DEFAULT_N_SIDE,
        n_bottom: int = _DEFAULT_N_BOTTOM,
        n_length: int = _DEFAULT_N_LENGTH,
    ) -> None:
        if any(v < 1 for v in (n_top, n_side, n_bottom, n_length)):
            raise ValueError(
                f"n_top ({n_top}), n_side ({n_side}), n_bottom ({n_bottom}), "
                f"n_length ({n_length}) must all be ≥ 1"
            )
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self._sec      = section
        self._L        = length
        self._n_top    = n_top
        self._n_side   = n_side
        self._n_bottom = n_bottom
        self._n_length = n_length

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> BeamSurfaceMesh:
        """Build and return the BeamSurfaceMesh for the I-profile beam."""
        s = self._sec
        L = self._L

        z_top_in  = s.h / 2.0 - s.tf_top   # inner face of top flange = web top junction
        z_bot_in  = -s.h / 2.0 + s.tf_bot  # inner face of bot flange = web bot junction
        yw_l      = -s.tw / 2.0             # left web face y (outward normal = −y)

        xs = np.linspace(0.0, L, self._n_length + 1)

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

        def add_face_xy(ys: np.ndarray, z_const: float, thickness: float) -> None:
            """Tile a horizontal face at z=z_const with (n_length) × (len(ys)-1) quads."""
            for ix in range(self._n_length):
                for iy in range(len(ys) - 1):
                    n0 = gid(xs[ix],     ys[iy],     z_const)
                    n1 = gid(xs[ix + 1], ys[iy],     z_const)
                    n2 = gid(xs[ix + 1], ys[iy + 1], z_const)
                    n3 = gid(xs[ix],     ys[iy + 1], z_const)
                    quads.append((n0, n1, n2, n3))
                    thicknesses.append(thickness)
                    local_cols.append((0, 1))  # FEM in (x, y)

        def add_face_xz(zs: np.ndarray, y_const: float, thickness: float) -> None:
            """Tile a vertical face at y=y_const with (n_length) × (len(zs)-1) quads."""
            for ix in range(self._n_length):
                for iz in range(len(zs) - 1):
                    n0 = gid(xs[ix],     y_const, zs[iz])
                    n1 = gid(xs[ix + 1], y_const, zs[iz])
                    n2 = gid(xs[ix + 1], y_const, zs[iz + 1])
                    n3 = gid(xs[ix],     y_const, zs[iz + 1])
                    quads.append((n0, n1, n2, n3))
                    thicknesses.append(thickness)
                    local_cols.append((0, 2))  # FEM in (x, z)

        # ── Top flange: inner face at z_top_in ────────────────────────────────
        # _flange_ys forces y=yw_l into the grid so gid() deduplicates the web
        # top-corner node (x_i, yw_l, z_top_in) with this face → shared DOF.
        ys_top = _flange_ys(s.bf_top, self._n_top, yw_l)
        add_face_xy(ys_top, z_top_in, s.tf_top)

        # ── Web — left face at y=yw_l, spanning z_bot_in to z_top_in ─────────
        # Per §3.4.1 the web is ONE plate receiving fire from both faces
        # (n_exposed_sides=2 in the solver).  Left face at y=yw_l gives outward
        # normal (0, −1, 0) — non-zero and correct for RadiationBall flux.
        zs_web = np.linspace(z_bot_in, z_top_in, self._n_side + 1)
        add_face_xz(zs_web, yw_l, s.tw)

        # ── Bottom flange: inner face at z_bot_in ─────────────────────────────
        # y=yw_l in ys_bot → shared node with web bottom-corner node.
        ys_bot = _flange_ys(s.bf_bot, self._n_bottom, yw_l)
        add_face_xy(ys_bot, z_bot_in, s.tf_bot)

        return BeamSurfaceMesh(
            nodes=np.array(pool, dtype=float),
            quads=np.array(quads, dtype=np.intp),
            thicknesses=np.array(thicknesses, dtype=float),
            local_cols=np.array(local_cols, dtype=np.intp),
        )


# ── Module helpers ────────────────────────────────────────────────────────────

def _snap(v: float) -> float:
    return round(v / _DEDUP_TOL) * _DEDUP_TOL


def _flange_ys(bf: float, n: int, y_jct: float) -> np.ndarray:
    """
    y-node positions for a flange of width *bf* divided into *n* elements,
    with *y_jct* (web T-junction) always present regardless of n parity.

    The flange is split at y_jct: left part (−bf/2 → y_jct) gets ⌈n × frac⌉
    elements and right part (y_jct → +bf/2) gets the remainder.  At least one
    element per side is guaranteed.  Total element count = n.
    """
    frac_left = (y_jct - (-bf / 2.0)) / bf   # fraction of width to the left of junction
    n_left    = min(n - 1, max(1, round(n * frac_left)))
    n_right   = n - n_left
    left  = np.linspace(-bf / 2.0, y_jct,      n_left  + 1)
    right = np.linspace(y_jct,     bf / 2.0,   n_right + 1)
    return np.concatenate([left, right[1:]])   # remove duplicate at y_jct
