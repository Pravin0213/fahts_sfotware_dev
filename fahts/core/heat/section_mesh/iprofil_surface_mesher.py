"""
I-profile (IHPROFIL) 3-D surface mesh generator (FAHTS axial × hoop approach).

Creates Quad4 shell elements on all exposed outer faces of an I/H-beam.
The 2-D heat equation is solved in the axial × hoop plane of each face element;
wall thickness is a scalar parameter (SINTEF FAHTS §3.2.2).

Default mesh divisions (matching FAHTS software defaults):
    n_top    = 4   elements across top flange width (hoop direction)
    n_side   = 2   elements along web height (hoop direction)
    n_bottom = 2   elements across bottom flange width (hoop direction)
    n_length = 2   elements along beam axis

Exposed face layout (beam-local: x = axial, y = hoop-horizontal, z = vertical):

  z = +h/2          ┌─────── top face (n_top × n_length) ────────┐
                    │        top flange, thickness = tf_top      │
  z = z_top_in  └─ ovhg_L ─┬── web top ──┬─ ovhg_R ─┘
                             │  left web   │
                             │  (n_side    │
  z = z_bot_in  ┌─ ovhg_L ─┴── web bot ──┴─ ovhg_R ─┐
                    │        bottom flange             │
  z = -h/2          └─────── bottom face (n_bottom × n_length) ──┘

ovhg_L / ovhg_R = overhang underside faces (horizontal, same thickness as flange).
Overhang element count is scaled proportionally to the flange element count.

Corner nodes at (x_i, ±tw/2, z_top_in) and (x_i, ±tw/2, z_bot_in) are shared
between the web and overhang faces via 3-D coordinate deduplication.
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import ISection
from .beam_surface_mesh import BeamSurfaceMesh

_DEDUP_TOL = 1e-10


class IProfileSurfaceMesher:
    """
    Generates the FAHTS-style surface Quad4 mesh for an I/H-profile beam.

    Args:
        section:   ISection to mesh.
        length:    Beam element length [m].
        n_top:     Elements across top flange width (default 4).
        n_side:    Elements along web height — each web face (default 2).
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

        z_top_out = s.h / 2.0              # top of top flange
        z_top_in  = z_top_out - s.tf_top   # underside of top flange
        z_bot_in  = -s.h / 2.0 + s.tf_bot # topside of bottom flange
        z_bot_out = -s.h / 2.0            # bottom of bottom flange

        yw_l = -s.tw / 2.0                # left web face y
        yw_r =  s.tw / 2.0                # right web face y

        xs = np.linspace(0.0, L, self._n_length + 1)

        # Global node pool
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

        # ── Top flange top face (full width) ──────────────────────────────────
        ys_top = np.linspace(-s.bf_top / 2.0, s.bf_top / 2.0, self._n_top + 1)
        add_face_xy(ys_top, z_top_out, s.tf_top)

        # ── Top flange underside — left and right overhangs ───────────────────
        ovhg_top = max(0.0, (s.bf_top - s.tw) / 2.0)
        if ovhg_top > 0.0:
            n_oh_top = max(1, round(self._n_top * ovhg_top / s.bf_top))
            ys_oh_L = np.linspace(-s.bf_top / 2.0, yw_l, n_oh_top + 1)
            ys_oh_R = np.linspace(yw_r, s.bf_top / 2.0, n_oh_top + 1)
            add_face_xy(ys_oh_L, z_top_in, s.tf_top)
            add_face_xy(ys_oh_R, z_top_in, s.tf_top)

        # ── Web — left and right faces ────────────────────────────────────────
        zs_web = np.linspace(z_bot_in, z_top_in, self._n_side + 1)
        add_face_xz(zs_web, yw_l, s.tw)
        add_face_xz(zs_web, yw_r, s.tw)

        # ── Bottom flange topside — left and right overhangs ──────────────────
        ovhg_bot = max(0.0, (s.bf_bot - s.tw) / 2.0)
        if ovhg_bot > 0.0:
            n_oh_bot = max(1, round(self._n_bottom * ovhg_bot / s.bf_bot))
            ys_oh_bL = np.linspace(-s.bf_bot / 2.0, yw_l, n_oh_bot + 1)
            ys_oh_bR = np.linspace(yw_r, s.bf_bot / 2.0, n_oh_bot + 1)
            add_face_xy(ys_oh_bL, z_bot_in, s.tf_bot)
            add_face_xy(ys_oh_bR, z_bot_in, s.tf_bot)

        # ── Bottom flange bottom face (full width) ────────────────────────────
        ys_bot = np.linspace(-s.bf_bot / 2.0, s.bf_bot / 2.0, self._n_bottom + 1)
        add_face_xy(ys_bot, z_bot_out, s.tf_bot)

        return BeamSurfaceMesh(
            nodes=np.array(pool, dtype=float),
            quads=np.array(quads, dtype=np.intp),
            thicknesses=np.array(thicknesses, dtype=float),
            local_cols=np.array(local_cols, dtype=np.intp),
        )


# ── Module helper ─────────────────────────────────────────────────────────────

def _snap(v: float) -> float:
    return round(v / _DEDUP_TOL) * _DEDUP_TOL
