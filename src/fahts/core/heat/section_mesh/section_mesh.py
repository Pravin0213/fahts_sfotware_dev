"""Data container for a 2-D FEM cross-section mesh."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SectionMesh:
    """
    2-D FEM mesh of a steel cross-section in the local y-z plane.

    Coordinate frame (beam cross-section):
        y-axis → width direction  (local y, i.e. W for BOX)
        z-axis → height direction (local z, i.e. H for BOX)
        Origin at section centroid.

    Attributes:
        nodes:            (n_nodes, 2)  — [y, z] coordinates [m]
        quads:            (n_quads, 4)  — node indices, CCW order
        outer_edge_pairs: (n_outer, 2) — edges on the exposed outer surface (Robin BC)
        inner_edge_pairs: (n_inner, 2) — edges on the inner hollow surface (adiabatic)
    """
    nodes: np.ndarray
    quads: np.ndarray
    outer_edge_pairs: np.ndarray
    inner_edge_pairs: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_quads(self) -> int:
        return len(self.quads)

    @property
    def outer_perimeter(self) -> float:
        """Sum of outer boundary edge lengths [m]."""
        return float(sum(
            np.linalg.norm(self.nodes[j] - self.nodes[i])
            for i, j in self.outer_edge_pairs
        ))

    @property
    def inner_perimeter(self) -> float:
        """Sum of inner hollow boundary edge lengths [m]."""
        return float(sum(
            np.linalg.norm(self.nodes[j] - self.nodes[i])
            for i, j in self.inner_edge_pairs
        ))

    @property
    def inner_node_indices(self) -> list[int]:
        """Unique node indices that lie on the inner hollow boundary."""
        return sorted({int(n) for pair in self.inner_edge_pairs for n in pair})

    @property
    def steel_area(self) -> float:
        """Total meshed steel area [m²] computed from quad Jacobians."""
        total = 0.0
        for quad in self.quads:
            y = self.nodes[quad, 0]
            z = self.nodes[quad, 1]
            # Shoelace formula for a (possibly non-planar) quad:
            # area = 0.5 * |d1 × d2| where d1,d2 are diagonals — but for
            # axis-aligned quads the simple (dy * dz) formula is exact.
            # Use cross-product of triangle pair for general quads.
            dy1 = y[2] - y[0]
            dz1 = z[2] - z[0]
            dy2 = y[3] - y[1]
            dz2 = z[3] - z[1]
            total += 0.5 * abs(dy1 * dz2 - dy2 * dz1)
        return total
