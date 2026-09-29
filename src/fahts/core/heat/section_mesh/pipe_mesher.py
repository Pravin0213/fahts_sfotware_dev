"""
Circular hollow (PIPE) cross-section 2-D FEM mesh generator (Quad4 elements).

Polar annular mesh in the y-z plane:
  r ∈ [R_in, R_out]   — n_radial layers (default 1)
  θ ∈ [0, 2π)         — n_arc divisions (default 8, matching USFOS default)

Outer edges (r = R_out): Robin BC — fire exposed.
Inner edges (r = R_in):  Adiabatic — hollow interior.
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import PipeSection
from .section_mesh import SectionMesh


class PipeMesher:
    """
    Structured Quad4 mesh generator for a circular hollow PIPE cross-section.

    Args:
        section:  PipeSection to mesh.
        n_arc:    Number of elements around the circumference (default 8).
        n_radial: Number of element layers through the wall thickness (default 1).
    """

    def __init__(
        self,
        section: PipeSection,
        n_arc: int = 8,
        n_radial: int = 1,
    ) -> None:
        if n_arc < 3:
            raise ValueError(f"n_arc must be >= 3, got {n_arc}")
        if n_radial < 1:
            raise ValueError(f"n_radial must be >= 1, got {n_radial}")
        if section.inner_radius <= 0:
            raise ValueError(
                f"PIPE {section.sid} has zero or negative inner radius — cannot mesh."
            )
        self._sec = section
        self._na = n_arc
        self._nr = n_radial

    def build(self) -> SectionMesh:
        s = self._sec
        na, nr = self._na, self._nr
        R_in  = s.inner_radius
        R_out = s.outer_radius

        # ── Node grid ─────────────────────────────────────────────────────────
        # node_idx[i, j] = flat index; i = radial level (0=inner, nr=outer), j = arc
        radii = np.linspace(R_in, R_out, nr + 1)
        angles = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)

        n_nodes = (nr + 1) * na
        nodes = np.zeros((n_nodes, 2))
        node_idx = np.zeros((nr + 1, na), dtype=np.intp)

        for i in range(nr + 1):
            for j in range(na):
                flat = i * na + j
                node_idx[i, j] = flat
                nodes[flat, 0] = radii[i] * np.cos(angles[j])   # y
                nodes[flat, 1] = radii[i] * np.sin(angles[j])   # z

        # ── Quads (CCW in y-z plane) ──────────────────────────────────────────
        quads: list[tuple[int, int, int, int]] = []
        for i in range(nr):
            for j in range(na):
                j1 = (j + 1) % na
                n00 = int(node_idx[i,     j])
                n10 = int(node_idx[i + 1, j])
                n11 = int(node_idx[i + 1, j1])
                n01 = int(node_idx[i,     j1])

                # Verify CCW: cross product of (v1-v0) x (v2-v0) should be +x
                v0 = nodes[n00]
                v1 = nodes[n10]
                v2 = nodes[n11]
                cross_x = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
                if cross_x >= 0:
                    quads.append((n00, n10, n11, n01))
                else:
                    quads.append((n00, n01, n11, n10))

        quads_arr = np.array(quads, dtype=np.intp)

        # ── Outer edges (r = R_out, i = nr) ──────────────────────────────────
        outer: list[tuple[int, int]] = []
        for j in range(na):
            j1 = (j + 1) % na
            outer.append((int(node_idx[nr, j]), int(node_idx[nr, j1])))

        # ── Inner edges (r = R_in, i = 0) — adiabatic ────────────────────────
        inner: list[tuple[int, int]] = []
        for j in range(na):
            j1 = (j + 1) % na
            inner.append((int(node_idx[0, j]), int(node_idx[0, j1])))

        return SectionMesh(
            nodes=nodes,
            quads=quads_arr,
            outer_edge_pairs=np.array(outer, dtype=np.intp),
            inner_edge_pairs=np.array(inner, dtype=np.intp),
        )
