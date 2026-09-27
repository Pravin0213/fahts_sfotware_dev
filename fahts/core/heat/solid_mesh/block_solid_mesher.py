"""
Rectangular block Hex8 solid mesher (analytical validation cases).

Node numbering is structured:
    node id = i + (nx + 1) · (j + (ny + 1) · k),   i ∈ [0, nx], j ∈ [0, ny], k ∈ [0, nz]
    node position = origin + (i · lx/nx, j · ly/ny, k · lz/nz)
Hex (i, j, k) has id = i + nx · (j + ny · k) and nodes
    [n(i,j,k), n(i+1,j,k), n(i+1,j+1,k), n(i,j+1,k),  (same at k+1)]
All six outer faces are FACE_OUTER.
"""
from __future__ import annotations

import numpy as np

from ._hex_topology import boundary_faces
from .solid_mesh import FACE_OUTER, SolidMesh


class BlockSolidMesher:
    """
    Hex8 mesh of an axis-aligned rectangular block.

    Args:
        lx, ly, lz: Block edge lengths [m] (> 0).
        nx, ny, nz: Elements per direction (≥ 1).
        origin:     Minimum corner (default (0, 0, 0)).
    """

    def __init__(
        self,
        lx: float, ly: float, lz: float,
        nx: int, ny: int, nz: int,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if min(lx, ly, lz) <= 0.0:
            raise ValueError("lx, ly, lz must be positive")
        if min(nx, ny, nz) < 1:
            raise ValueError("nx, ny, nz must be >= 1")
        self._l = (float(lx), float(ly), float(lz))
        self._n = (int(nx), int(ny), int(nz))
        self._origin = np.asarray(origin, dtype=float)

    def node_id(self, i: int, j: int, k: int) -> int:
        """Structured node id of grid point (i, j, k)."""
        nx, ny, _ = self._n
        return i + (nx + 1) * (j + (ny + 1) * k)

    def build(self) -> SolidMesh:
        """Build and return the validated SolidMesh."""
        nx, ny, nz = self._n
        lx, ly, lz = self._l
        xs = np.linspace(0.0, lx, nx + 1)
        ys = np.linspace(0.0, ly, ny + 1)
        zs = np.linspace(0.0, lz, nz + 1)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")    # i fastest
        nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) + self._origin

        K, J, I = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        n0 = (I + (nx + 1) * (J + (ny + 1) * K)).ravel()
        dy, dz = nx + 1, (nx + 1) * (ny + 1)
        bottom = np.stack([n0, n0 + 1, n0 + 1 + dy, n0 + dy], axis=1)
        hexes = np.concatenate([bottom, bottom + dz], axis=1).astype(np.intp)

        faces, _, _ = boundary_faces(hexes)
        group = np.full(len(faces), FACE_OUTER, dtype=np.intp)
        mesh = SolidMesh(nodes=nodes, hexes=hexes, faces=faces, face_group=group,
                         section_kind="BLOCK")
        mesh.validate()
        return mesh
