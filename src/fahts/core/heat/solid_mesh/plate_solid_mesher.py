"""
Flat plate (QUADSHEL) Hex8 solid mesher — global coordinates.

The mid-surface is the bilinear grid of PlateSurfaceMesher (mesh_12 elements
along n1→n2, mesh_14 along n1→n4).  It is offset along the plate normal
n = normalise((P2 − P1) × (P4 − P1)) to n_layers + 1 levels
    offset = −t/2 + l · t / n_layers,  l = 0 … n_layers,
so the solid occupies mid-surface ± t/2.  Node id = l · (m12+1)(m14+1)
+ i · (m12 + 1) + j  (i along n1→n4, j along n1→n2).

Faces: both large faces (−n side and +n side) → FACE_OUTER; the four thickness
edges → FACE_END (internal / connected), unless flagged in ``exposed_edges`` (free
plate edges, which receive fire / source heat on their thickness faces) → FACE_OUTER.
"""
from __future__ import annotations

import numpy as np

from fahts.core.model.section import PlateSection
from ._hex_topology import LOCAL_FACE_ZETA_MINUS, LOCAL_FACE_ZETA_PLUS, boundary_faces
from .solid_mesh import FACE_END, FACE_OUTER, SolidMesh


class PlateSolidMesher:
    """
    Hex8 solid mesh of a flat plate element.

    Args:
        section:  PlateSection (thickness).
        corners:  (4, 3) global positions of QUADSHEL nodes n1…n4 [m].
        mesh_12:  Elements along n1 → n2 (default 4).
        mesh_14:  Elements along n1 → n4 (default 2).
        n_layers: Elements through the thickness (default 2).
        exposed_edges: 4 flags for the edges n1–n2, n2–n3, n3–n4, n4–n1; True makes that
                  edge's thickness faces FACE_OUTER (free edge).  Default: all internal.
    """

    def __init__(
        self,
        section: PlateSection,
        corners: np.ndarray,
        mesh_12: int = 4,
        mesh_14: int = 2,
        n_layers: int = 2,
        exposed_edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
    ) -> None:
        corners = np.asarray(corners, dtype=float)
        if corners.shape != (4, 3):
            raise ValueError(f"corners must be shape (4, 3), got {corners.shape}")
        if min(mesh_12, mesh_14, n_layers) < 1:
            raise ValueError("mesh_12, mesh_14 and n_layers must be >= 1")
        if section.thickness <= 0.0:
            raise ValueError(f"plate thickness must be positive, got {section.thickness}")
        self._sec = section
        self._corners = corners
        self._m12 = mesh_12
        self._m14 = mesh_14
        self._n_layers = n_layers
        if len(exposed_edges) != 4:
            raise ValueError("exposed_edges needs 4 flags (edges 12, 23, 34, 41)")
        self._exposed_edges = tuple(bool(x) for x in exposed_edges)

    @property
    def normal(self) -> np.ndarray:
        """Unit plate normal (P2 − P1) × (P4 − P1), direction of increasing layer."""
        P1, P2, _, P4 = self._corners
        n = np.cross(P2 - P1, P4 - P1)
        mag = np.linalg.norm(n)
        if mag < 1e-12:
            raise ValueError("Degenerate plate element: nodes are collinear or too close.")
        return n / mag

    def build(self) -> SolidMesh:
        """Build and return the validated SolidMesh."""
        P1, P2, P3, P4 = self._corners
        m12, m14, nl = self._m12, self._m14, self._n_layers
        t = self._sec.thickness
        n = self.normal

        xi = np.arange(m12 + 1) / m12                       # along n1→n2 (index j)
        eta = np.arange(m14 + 1) / m14                      # along n1→n4 (index i)
        X, E = np.meshgrid(xi, eta)                         # (m14+1, m12+1)
        X, E = X[..., None], E[..., None]
        mid = ((1 - X) * (1 - E) * P1 + X * (1 - E) * P2
               + X * E * P3 + (1 - X) * E * P4).reshape(-1, 3)
        offs = -t / 2.0 + t * np.arange(nl + 1) / nl
        nodes = (mid[None, :, :] + offs[:, None, None] * n[None, None, :]).reshape(-1, 3)

        n_mid = (m12 + 1) * (m14 + 1)
        i, j = np.meshgrid(np.arange(m14), np.arange(m12), indexing="ij")
        i, j = i.ravel(), j.ravel()
        n00 = i * (m12 + 1) + j
        # CCW seen from +n: (n1→n2 direction) × (n1→n4 direction) = +n
        q = np.stack([n00, n00 + 1, n00 + m12 + 2, n00 + m12 + 1], axis=1)
        layer = np.arange(nl)[:, None, None] * n_mid
        base = q[None] + layer
        hexes = np.concatenate([base, base + n_mid], axis=2).reshape(-1, 8)

        faces, _, local_face = boundary_faces(hexes)
        big = (local_face == LOCAL_FACE_ZETA_MINUS) | (local_face == LOCAL_FACE_ZETA_PLUS)
        group = np.where(big, FACE_OUTER, FACE_END).astype(np.intp)
        if any(self._exposed_edges):
            # grid position of every face node: i along n1→n4, j along n1→n2
            g = faces % n_mid
            fi, fj = g // (m12 + 1), g % (m12 + 1)
            on_edge = (
                (fi == 0).all(axis=1),          # n1–n2
                (fj == m12).all(axis=1),        # n2–n3
                (fi == m14).all(axis=1),        # n3–n4
                (fj == 0).all(axis=1),          # n4–n1
            )
            for flag, sel in zip(self._exposed_edges, on_edge):
                if flag:
                    group[sel & ~big] = FACE_OUTER
        mesh = SolidMesh(nodes=nodes, hexes=hexes.astype(np.intp), faces=faces,
                         face_group=group, section_kind="PLATE")
        mesh.validate()
        return mesh
