"""
I-profile (IHPROFIL) cross-section 2-D FEM mesh generator (Quad4 elements).

Layout (y = width direction, z = height direction, origin at centroid):

  z = +h/2  ┌─────────────────────────────────┐  top flange outer top
             │         top flange              │  tf_top thick, bf_top wide
  z=+h/2-tf_top └────┬──────────────────┬────┘
                     │     web          │  tw wide
                     │                  │  web_height = h - tf_top - tf_bot
  z=-h/2+tf_bot ┌────┴──────────────────┴────┐
             │         bottom flange          │  tf_bot thick, bf_bot wide
  z = -h/2  └─────────────────────────────────┘  bottom flange outer bottom

All three parts share nodes at the flange-web junctions via coordinate dedup.
No hollow interior: inner_edge_pairs is always empty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from fahts.core.model.section import ISection
from .section_mesh import SectionMesh

_DEDUP_TOL = 1e-10


class IProfileMesher:
    """
    Structured Quad4 mesh for an I/H-profile cross-section.

    Args:
        section:   The ISection to mesh.
        elem_size: Target element size along flanges and web [m].
                   Defaults to min(tf_top, tf_bot, tw).
        n_layers:  Number of Quad4 layers through the flange/web thickness (≥ 1).
    """

    def __init__(
        self,
        section: ISection,
        elem_size: Optional[float] = None,
        n_layers: int = 1,
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if section.web_height <= 0:
            raise ValueError(
                f"ISection {section.sid} has non-positive web height ({section.web_height:.4f} m)."
            )
        self._sec = section
        self._n = n_layers
        self._es = (
            elem_size if elem_size is not None
            else min(section.tf_top, section.tf_bot, section.tw)
        )

    def build(self) -> SectionMesh:
        s = self._sec
        n, es = self._n, self._es

        # Key z coordinates
        z_bot_out = -s.h / 2.0
        z_bot_in  = z_bot_out + s.tf_bot
        z_top_in  = s.h / 2.0 - s.tf_top
        z_top_out = s.h / 2.0

        # Key y coordinates
        y_fl_top_l, y_fl_top_r = -s.bf_top / 2.0, s.bf_top / 2.0
        y_fl_bot_l, y_fl_bot_r = -s.bf_bot / 2.0, s.bf_bot / 2.0
        y_web_l,    y_web_r    = -s.tw / 2.0,      s.tw / 2.0

        # ── Node pool with deduplication ──────────────────────────────────────
        pool: list[np.ndarray] = []
        idx: dict[tuple[float, float], int] = {}

        def gid(y: float, z: float) -> int:
            key = (_snap(y), _snap(z))
            if key not in idx:
                idx[key] = len(pool)
                pool.append(np.array([y, z], dtype=float))
            return idx[key]

        quads: list[tuple[int, int, int, int]] = []

        def mesh_patch(ys: np.ndarray, zs: np.ndarray) -> None:
            for j in range(len(zs) - 1):
                for i in range(len(ys) - 1):
                    quads.append((
                        gid(ys[i],     zs[j]),
                        gid(ys[i + 1], zs[j]),
                        gid(ys[i + 1], zs[j + 1]),
                        gid(ys[i],     zs[j + 1]),
                    ))

        # ── Top flange ────────────────────────────────────────────────────────
        # y: left overhang | web width | right overhang
        # z: n layers through tf_top
        y_fl_top_L = np.linspace(y_fl_top_l, y_web_l,  max(2, _n_divs(y_web_l - y_fl_top_l, es) + 1))
        y_fl_top_M = np.linspace(y_web_l,    y_web_r,  n + 1)
        y_fl_top_R = np.linspace(y_web_r,    y_fl_top_r, max(2, _n_divs(y_fl_top_r - y_web_r, es) + 1))
        y_top_plate = _unique_1d([y_fl_top_L, y_fl_top_M, y_fl_top_R])
        z_top_plate = np.linspace(z_top_in, z_top_out, n + 1)
        mesh_patch(y_top_plate, z_top_plate)

        # ── Bottom flange ─────────────────────────────────────────────────────
        y_fl_bot_L = np.linspace(y_fl_bot_l, y_web_l,  max(2, _n_divs(y_web_l - y_fl_bot_l, es) + 1))
        y_fl_bot_M = np.linspace(y_web_l,    y_web_r,  n + 1)
        y_fl_bot_R = np.linspace(y_web_r,    y_fl_bot_r, max(2, _n_divs(y_fl_bot_r - y_web_r, es) + 1))
        y_bot_plate = _unique_1d([y_fl_bot_L, y_fl_bot_M, y_fl_bot_R])
        z_bot_plate = np.linspace(z_bot_out, z_bot_in, n + 1)
        mesh_patch(y_bot_plate, z_bot_plate)

        # ── Web ───────────────────────────────────────────────────────────────
        y_web = np.linspace(y_web_l, y_web_r, n + 1)
        z_web = np.linspace(z_bot_in, z_top_in, max(2, _n_divs(s.web_height, es) + 1))
        mesh_patch(y_web, z_web)

        nodes_arr = np.array(pool, dtype=float)
        quads_arr = np.array(quads, dtype=np.intp)

        outer = _all_boundary_edges(quads_arr)

        return SectionMesh(
            nodes=nodes_arr,
            quads=quads_arr,
            outer_edge_pairs=(
                np.array(outer, dtype=np.intp) if outer
                else np.empty((0, 2), dtype=np.intp)
            ),
            inner_edge_pairs=np.empty((0, 2), dtype=np.intp),
        )


# ── Module helpers ────────────────────────────────────────────────────────────

def _snap(v: float) -> float:
    return round(v / _DEDUP_TOL) * _DEDUP_TOL


def _n_divs(length: float, elem_size: float) -> int:
    return max(1, round(abs(length) / elem_size))


def _unique_1d(arrays: list[np.ndarray]) -> np.ndarray:
    seen: set[float] = set()
    out: list[float] = []
    for arr in arrays:
        for v in arr.tolist():
            k = round(v, 12)
            if k not in seen:
                seen.add(k)
                out.append(v)
    return np.array(out, dtype=float)


def _all_boundary_edges(quads: np.ndarray) -> list[tuple[int, int]]:
    """Return all single-occurrence boundary edges (directed, CCW order)."""
    edge_count: dict[tuple[int, int], int] = {}
    edge_dir: dict[tuple[int, int], tuple[int, int]] = {}
    for quad in quads:
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_dir:
                edge_dir[key] = (a, b)
    return [edge_dir[key] for key, cnt in edge_count.items() if cnt == 1]
