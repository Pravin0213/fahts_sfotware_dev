"""
BeamSurfaceMesh — 3-D shell-element surface mesh for the FAHTS beam approach.

Each 1-D structural beam element is re-meshed into Quad4 shell elements that
tile the outer faces of the section.  The 2-D heat equation is solved in the
axial × hoop plane of each element; wall thickness is a scalar parameter,
not a mesh dimension (SINTEF FAHTS §3.2.2, Eq. 3.2.2).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BeamSurfaceMesh:
    """
    Quad4 shell elements covering the outer surface of a beam.

    Coordinate frame (beam-local):
        x-axis  beam axis (longitudinal), x ∈ [0, beam_length]
        y-axis  cross-section width direction
        z-axis  cross-section height direction
        Origin at the start node of the beam element.

    Attributes:
        nodes:           (n_nodes, 3) — node [x, y, z] coordinates [m]
        quads:           (n_quads, 4) — node indices, CCW from outside
        thicknesses:     (n_quads,)   — wall thickness [m] per element
        local_cols:      (n_quads, 2) int — column indices into nodes[:, ?] for
                         the local 2-D FEM coordinates:
                         [0, 1] → (x, y) for top/bottom faces
                         [0, 2] → (x, z) for left/right faces
        local_coords_2d: (n_quads, 4, 2) optional — explicit pre-computed 2-D
                         coordinates for each element, used when local_cols is
                         not applicable (e.g. cylindrical PIPE surface or
                         arbitrarily oriented plate elements).  When set,
                         element_coords_2d() returns these instead of extracting
                         from nodes via local_cols.
    """
    nodes: np.ndarray            # (n_nodes, 3)
    quads: np.ndarray            # (n_quads, 4)
    thicknesses: np.ndarray      # (n_quads,)
    local_cols: np.ndarray       # (n_quads, 2)
    local_coords_2d: np.ndarray | None = field(default=None, repr=False)  # (n_quads, 4, 2)
    # Quad indices exposed to fire; empty = all quads (default for backward compat)
    outer_face_indices: list[int] = field(default_factory=list)
    # Node indices on the sealed inner surface (BOX/PIPE hollow cavity); empty = none
    inner_node_indices: list[int] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_quads(self) -> int:
        return len(self.quads)

    def element_coords_2d(self, q: int) -> np.ndarray:
        """
        Local 2-D node coordinates for element q, shape (4, 2).

        Returns local_coords_2d[q] when the explicit array is set (PIPE,
        shell plates); otherwise extracts columns from the 3-D node array
        via local_cols (BOX, I-profile).
        """
        if self.local_coords_2d is not None:
            return self.local_coords_2d[q]
        return self.nodes[self.quads[q]][:, self.local_cols[q]]

    @property
    def node_area_weights(self) -> np.ndarray:
        """
        (n_nodes,) normalised area weights for computing area-weighted mean T.

        Each node accumulates 1/4 of every attached element's face area.
        """
        node_areas = np.zeros(self.n_nodes)
        for q, quad in enumerate(self.quads):
            coords = self.element_coords_2d(q)
            # Diagonal cross-product area formula (exact for axis-aligned quads)
            d1 = coords[2] - coords[0]
            d2 = coords[3] - coords[1]
            A_e = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
            for nid in quad:
                node_areas[int(nid)] += A_e * 0.25
        total = node_areas.sum()
        return node_areas / total if total > 0.0 else np.ones(self.n_nodes) / self.n_nodes
