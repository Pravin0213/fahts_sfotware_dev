"""
I/H-profile (IHPROFIL) Hex8 solid mesher.

Geometry follows IProfileSurfaceMesher's frame: beam-local x = axis ∈ [0, L],
y = flange width direction, z = height; the section is doubly centred on the
origin in y-z (outer flange faces at z = ±h/2, web y ∈ [−tw/2, +tw/2],
flanges centred on y = 0).  Unlike the surface mesher (mid/inner planes), the
solid mesher meshes the real steel volume:

    top flange    y ∈ [−bf_top/2, bf_top/2],  z ∈ [h/2 − tf_top, h/2]
    web           y ∈ [−tw/2, tw/2],          z ∈ [−h/2 + tf_bot, h/2 − tf_top]
    bottom flange y ∈ [−bf_bot/2, bf_bot/2],  z ∈ [−h/2, −h/2 + tf_bot]

Discretisation (each block a structured tensor grid, nodes shared by
coordinate deduplication so the web-flange junction is conforming):
    - n_layers elements through each flange thickness and across the web
      thickness; the flange strip above/below the web reuses the web's n_layers
      y-divisions,
    - flange outstands: n_top (top) / n_bottom (bottom) elements split between
      the left and right outstand (⌊n/2⌋ left, rest right, ≥ 1 each),
    - n_side elements along the clear web height.

All lateral boundary faces are FACE_OUTER (open section); x = 0 / L → FACE_END.
"""
from __future__ import annotations

import numpy as np

from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.model.section import ISection
from ._hex_topology import boundary_edges_2d, snap
from .extrude import extrude_section
from .solid_mesh import SolidMesh


class IProfileSolidMesher:
    """
    Hex8 solid mesh of an I/H-profile beam.

    Args:
        section:  ISection to mesh.
        length:   Member length [m].
        n_top:    Elements across the top-flange outstands (default 4).
        n_side:   Elements along the clear web height (default 2).
        n_bottom: Elements across the bottom-flange outstands (default 2).
        n_length: Axial elements (default 2).
        n_layers: Elements through flange / web thickness (default 2).
    """

    def __init__(
        self,
        section: ISection,
        length: float,
        n_top: int = 4,
        n_side: int = 2,
        n_bottom: int = 2,
        n_length: int = 2,
        n_layers: int = 2,
    ) -> None:
        if min(n_top, n_side, n_bottom, n_length, n_layers) < 1:
            raise ValueError("n_top, n_side, n_bottom, n_length, n_layers must be >= 1")
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        s = section
        if s.web_height <= 0.0 or s.bf_top <= s.tw or s.bf_bot <= s.tw:
            raise ValueError(
                f"IHPROFIL {s.sid}: need h > tf_top + tf_bot and flange widths > tw"
            )
        self._sec = section
        self._L = length
        self._n_top = n_top
        self._n_side = n_side
        self._n_bottom = n_bottom
        self._n_length = n_length
        self._n_layers = n_layers

    def build(self) -> SolidMesh:
        """Build and return the validated SolidMesh."""
        s = self._sec
        nl = self._n_layers
        z_top_in = s.h / 2.0 - s.tf_top
        z_bot_in = -s.h / 2.0 + s.tf_bot
        y_web = np.linspace(-s.tw / 2.0, s.tw / 2.0, nl + 1)

        pool: list[tuple[float, float]] = []
        idx: dict[tuple[float, float], int] = {}

        def gid(y: float, z: float) -> int:
            key = (snap(y), snap(z))
            if key not in idx:
                idx[key] = len(pool)
                pool.append((y, z))
            return idx[key]

        quads: list[tuple[int, int, int, int]] = []

        def patch(ys: np.ndarray, zs: np.ndarray) -> None:
            for jz in range(len(zs) - 1):
                for iy in range(len(ys) - 1):
                    quads.append((
                        gid(ys[iy], zs[jz]), gid(ys[iy + 1], zs[jz]),
                        gid(ys[iy + 1], zs[jz + 1]), gid(ys[iy], zs[jz + 1]),
                    ))

        patch(_flange_ys(s.bf_top, self._n_top, y_web),
              np.linspace(z_top_in, s.h / 2.0, nl + 1))
        patch(y_web, np.linspace(z_bot_in, z_top_in, self._n_side + 1))
        patch(_flange_ys(s.bf_bot, self._n_bottom, y_web),
              np.linspace(-s.h / 2.0, z_bot_in, nl + 1))

        quads_arr = np.array(quads, dtype=np.intp)
        sec2d = SectionMesh(
            nodes=np.array(pool, dtype=float),
            quads=quads_arr,
            outer_edge_pairs=boundary_edges_2d(quads_arr),
            inner_edge_pairs=np.empty((0, 2), dtype=np.intp),
        )
        return extrude_section(sec2d, self._L, self._n_length, section_kind="IPROFILE")


def _flange_ys(bf: float, n: int, y_web: np.ndarray) -> np.ndarray:
    """Flange y-grid: left outstand (⌊n/2⌋ ≥ 1), web strip, right outstand."""
    n_left = max(1, n // 2)
    n_right = max(1, n - n_left)
    left = np.linspace(-bf / 2.0, y_web[0], n_left + 1)
    right = np.linspace(y_web[-1], bf / 2.0, n_right + 1)
    return np.concatenate([left[:-1], y_web, right[1:]])
