"""
Circular hollow (PIPE) 3-D surface mesh generator (FAHTS axial × hoop approach).

Creates Quad4 shell elements on the outer cylindrical surface of a PIPE beam.
The 2-D heat equation is solved in the axial × hoop (arc-length) plane of each
element; wall thickness is a scalar parameter (SINTEF FAHTS §3.2.2).

Default mesh divisions (matching FAHTS software defaults):
    c_circ   = 8   elements around the circumference (hoop direction)
    n_length = 4   elements along beam axis

Node positions in beam-local 3-D space (x = axial):
    x_i = i · L / n_length   for i = 0 … n_length
    θ_j = j · 2π / c_circ    for j = 0 … c_circ−1

    nodes[i·c_circ + j] = (x_i, R·cos θ_j, R·sin θ_j)

2-D local coordinates for FEM (unrolled cylinder — axial × arc-length):
    For element [i, j]:
        node 0: (x_i,     j·R·Δθ)
        node 1: (x_{i+1}, j·R·Δθ)
        node 2: (x_{i+1}, (j+1)·R·Δθ)
        node 3: (x_i,     (j+1)·R·Δθ)

    where Δθ = 2π / c_circ.

    The last element (j = c_circ−1) wraps around to node index 0 in the 3-D
    array, but its local_coords_2d correctly spans arc length from
    (c_circ−1)·R·Δθ to 2π·R.

Stored in BeamSurfaceMesh.local_coords_2d — local_cols is unused (set to zeros).
"""
from __future__ import annotations

import math
import numpy as np

from fahts.core.model.section import PipeSection
from .beam_surface_mesh import BeamSurfaceMesh


class PipeSurfaceMesher:
    """
    Generates the FAHTS-style surface Quad4 mesh for a PIPE beam.

    Args:
        section:  PipeSection to mesh.
        length:   Beam element length [m].
        c_circ:   Elements around the circumference (default 8).
        n_length: Elements along the beam axis (default 4).
    """

    _DEFAULT_C_CIRC   = 8
    _DEFAULT_N_LENGTH = 4

    def __init__(
        self,
        section: PipeSection,
        length: float,
        c_circ: int = _DEFAULT_C_CIRC,
        n_length: int = _DEFAULT_N_LENGTH,
    ) -> None:
        if c_circ < 3:
            raise ValueError(f"c_circ must be >= 3, got {c_circ}")
        if n_length < 1:
            raise ValueError(f"n_length must be >= 1, got {n_length}")
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self._sec      = section
        self._L        = length
        self._c_circ   = c_circ
        self._n_length = n_length

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> BeamSurfaceMesh:
        """Build and return the BeamSurfaceMesh for the PIPE beam."""
        R = self._sec.outer_radius
        t = self._sec.thickness
        L = self._L
        nc = self._c_circ
        nl = self._n_length

        xs     = np.linspace(0.0, L, nl + 1)
        thetas = np.array([j * 2.0 * math.pi / nc for j in range(nc)])
        dtheta = 2.0 * math.pi / nc
        arc    = R * dtheta   # arc length per circumferential division [m]

        # ── 3-D node positions ────────────────────────────────────────────────
        # node_idx[i, j] = i * nc + j
        n_nodes = (nl + 1) * nc
        nodes   = np.zeros((n_nodes, 3))
        for i, xi in enumerate(xs):
            for j in range(nc):
                nodes[i * nc + j] = [xi, R * math.cos(thetas[j]),
                                          R * math.sin(thetas[j])]

        # ── Quads and local 2-D coordinates ──────────────────────────────────
        n_quads = nl * nc
        quads          = np.zeros((n_quads, 4), dtype=np.intp)
        thicknesses    = np.full(n_quads, t, dtype=float)
        local_cols_arr = np.zeros((n_quads, 2), dtype=np.intp)   # unused
        local_coords   = np.zeros((n_quads, 4, 2), dtype=float)  # (x, arc-length)

        q = 0
        for i in range(nl):
            for j in range(nc):
                j1 = (j + 1) % nc
                n00 = i       * nc + j
                n10 = (i + 1) * nc + j
                n11 = (i + 1) * nc + j1
                n01 = i       * nc + j1
                quads[q] = [n00, n10, n11, n01]

                # Unrolled 2-D coords: (axial, arc-length from start)
                x0, x1 = float(xs[i]), float(xs[i + 1])
                s0 = j * arc          # arc length of current circumferential line
                s1 = (j + 1) * arc    # arc length of next (wraps to full circumference)
                local_coords[q] = [
                    [x0, s0],
                    [x1, s0],
                    [x1, s1],
                    [x0, s1],
                ]
                q += 1

        return BeamSurfaceMesh(
            nodes=nodes,
            quads=quads,
            thicknesses=thicknesses,
            local_cols=local_cols_arr,
            local_coords_2d=local_coords,
        )
