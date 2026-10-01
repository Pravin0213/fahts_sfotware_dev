"""
SolidMesh — 3-D Hex8 solid mesh of a structural member (contract for the 3-D solver).

Every wall of the member is meshed as real volume: n_layers of 8-node bricks
through the wall thickness, so temperature may vary through the thickness,
around the section and along the member.

Hex8 local node ordering (VTK_HEXAHEDRON), natural coords (ξ, η, ζ):
    0 (−,−,−)   1 (+,−,−)   2 (+,+,−)   3 (−,+,−)
    4 (−,−,+)   5 (+,−,+)   6 (+,+,+)   7 (−,+,+)
Meshers MUST produce elements with a positive Jacobian determinant.

Boundary faces are Quad4 node quadruples ordered counter-clockwise when viewed
from OUTSIDE the solid, so (p2 − p0) × (p3 − p1) is the outward normal.

Face groups:
    FACE_OUTER  fire-exposed external surface (convection, radiation, sources)
    FACE_INNER  cavity / wetted surface of hollow members (adiabatic today;
                future hook for coupling to vessel contents)
    FACE_END    member end caps (adiabatic; joints couple via node merging)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACE_OUTER: int = 0
FACE_INNER: int = 1
FACE_END: int = 2

# 2-point Gauss rule on [−1, 1]
_G = 1.0 / np.sqrt(3.0)
_GP_1D = (-_G, _G)


def _quad_shape_derivs(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear Quad4 shape functions N (4,) and derivatives dN (2, 4)."""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
    ])
    dN = 0.25 * np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
    ])
    return N, dN


def _hex_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Hex8 shape-function derivatives dN (3, 8) w.r.t. (ξ, η, ζ)."""
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    dN = np.empty((3, 8))
    dN[0] = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dN[1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta)
    dN[2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
    return dN


@dataclass
class SolidMesh:
    """
    Hex8 solid mesh of one structural member.

    Coordinate frame: identical to BeamSurfaceMesh — beam-local for beam
    members (x = axis ∈ [0, L], y = width, z = height, origin at the start
    node incl. eccentricity), global for plate/shell members.

    Attributes:
        nodes:       (n_nodes, 3) node coordinates [m]
        hexes:       (n_hex, 8)   node indices, VTK Hex8 ordering, det J > 0
        faces:       (n_faces, 4) boundary Quad4 faces, CCW seen from outside
        face_group:  (n_faces,)   FACE_OUTER / FACE_INNER / FACE_END
        section_kind: "BOX" | "IPROFILE" | "PIPE" | "PLATE" | "BLOCK" (informational)
    """
    nodes: np.ndarray
    hexes: np.ndarray
    faces: np.ndarray
    face_group: np.ndarray
    section_kind: str = "BLOCK"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ── sizes ────────────────────────────────────────────────────────────────
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_hexes(self) -> int:
        return len(self.hexes)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    # ── face subsets ─────────────────────────────────────────────────────────
    def face_indices(self, group: int) -> np.ndarray:
        """Indices into `faces` belonging to one face group."""
        return np.flatnonzero(self.face_group == group)

    @property
    def outer_face_indices(self) -> np.ndarray:
        return self.face_indices(FACE_OUTER)

    @property
    def inner_face_indices(self) -> np.ndarray:
        return self.face_indices(FACE_INNER)

    @property
    def inner_node_indices(self) -> np.ndarray:
        """Unique node indices on the inner (cavity / wetted) surface."""
        idx = self.inner_face_indices
        if len(idx) == 0:
            return np.zeros(0, dtype=np.intp)
        return np.unique(self.faces[idx].ravel()).astype(np.intp)

    # ── face geometry ────────────────────────────────────────────────────────
    def face_centroids(self) -> np.ndarray:
        """(n_faces, 3) mean of the 4 corner nodes."""
        return self.nodes[self.faces].mean(axis=1)

    def face_normals(self) -> np.ndarray:
        """(n_faces, 3) unit outward normals from the diagonal cross product."""
        p = self.nodes[self.faces]
        n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
        mag = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(mag > 0.0, mag, 1.0)

    def face_areas(self) -> np.ndarray:
        """(n_faces,) exact area of each (possibly curved) bilinear face [m²]."""
        if "face_areas" not in self._cache:
            p = self.nodes[self.faces]                     # (n_f, 4, 3)
            area = np.zeros(len(self.faces))
            for xi in _GP_1D:
                for eta in _GP_1D:
                    _, dN = _quad_shape_derivs(xi, eta)
                    t1 = np.einsum("j,fjk->fk", dN[0], p)
                    t2 = np.einsum("j,fjk->fk", dN[1], p)
                    area += np.linalg.norm(np.cross(t1, t2), axis=1)
            self._cache["face_areas"] = area
        return self._cache["face_areas"]

    # ── volume geometry ──────────────────────────────────────────────────────
    def hex_volumes(self) -> np.ndarray:
        """(n_hex,) element volumes via 2×2×2 Gauss [m³] (signed; > 0 if valid)."""
        if "hex_volumes" not in self._cache:
            X = self.nodes[self.hexes]                     # (n_h, 8, 3)
            vol = np.zeros(len(self.hexes))
            for xi in _GP_1D:
                for eta in _GP_1D:
                    for zeta in _GP_1D:
                        dN = _hex_shape_derivs(xi, eta, zeta)
                        J = np.einsum("aj,hjk->hak", dN, X)
                        vol += np.linalg.det(J)
            self._cache["hex_volumes"] = vol
        return self._cache["hex_volumes"]

    @property
    def volume(self) -> float:
        return float(self.hex_volumes().sum())

    @property
    def node_volume_weights(self) -> np.ndarray:
        """(n_nodes,) normalised lumped-volume weights (1/8 of each hex per node)."""
        w = np.zeros(self.n_nodes)
        np.add.at(w, self.hexes.ravel(), np.repeat(self.hex_volumes() / 8.0, 8))
        total = w.sum()
        return w / total if total > 0.0 else np.full(self.n_nodes, 1.0 / self.n_nodes)

    def validate(self) -> None:
        """Raise ValueError on inverted hexes or inward-facing boundary faces."""
        if np.any(self.hex_volumes() <= 0.0):
            bad = np.flatnonzero(self.hex_volumes() <= 0.0)[:5]
            raise ValueError(f"Non-positive Hex8 volume in elements {bad.tolist()}")
        if len(self.face_group) != len(self.faces):
            raise ValueError("face_group length must equal number of faces")
