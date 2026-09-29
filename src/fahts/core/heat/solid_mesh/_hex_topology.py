"""
Private topology helpers shared by the Hex8 solid meshers.

Functions only (no classes): boundary-face extraction and 2-D cross-section
utilities used by `extrude_section` and the section-specific meshers.
"""
from __future__ import annotations

import numpy as np

# VTK Hex8 faces, each ordered CCW when seen from OUTSIDE a positive-Jacobian hex.
#   0: ζ = −1   1: ζ = +1   2: η = −1   3: ξ = +1   4: η = +1   5: ξ = −1
HEX_FACES: np.ndarray = np.array([
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
], dtype=np.intp)

LOCAL_FACE_ZETA_MINUS: int = 0
LOCAL_FACE_ZETA_PLUS: int = 1

_DEDUP_TOL = 1e-10


def boundary_faces(hexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the boundary Quad4 faces of a Hex8 mesh (faces used by exactly one hex).

    Returns:
        faces:      (n_b, 4) outward-oriented node quadruples
        hex_idx:    (n_b,)   owning hex index
        local_face: (n_b,)   local face index into HEX_FACES
    """
    hexes = np.asarray(hexes, dtype=np.intp)
    n_hex = len(hexes)
    all_faces = hexes[:, HEX_FACES].reshape(-1, 4)           # (6 n_hex, 4)
    keys = np.sort(all_faces, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    once = counts[inv] == 1
    sel = np.flatnonzero(once)
    return all_faces[sel], sel // 6, sel % 6


def snap(v: float) -> float:
    """Round a coordinate onto a 1e-10 m lattice for dictionary deduplication."""
    return round(v / _DEDUP_TOL) * _DEDUP_TOL


def boundary_edges_2d(quads: np.ndarray) -> np.ndarray:
    """(n_e, 2) directed boundary edges of a 2-D quad mesh (edges used once)."""
    quads = np.asarray(quads, dtype=np.intp)
    e = np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2)
    keys = np.sort(e, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    return e[counts[inv] == 1]


def quad_signed_areas_2d(nodes: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(n_q,) signed shoelace areas of 2-D quads (> 0 for CCW)."""
    p = np.asarray(nodes, dtype=float)[np.asarray(quads, dtype=np.intp)]
    y, z = p[..., 0], p[..., 1]
    return 0.5 * np.sum(y * np.roll(z, -1, axis=1) - np.roll(y, -1, axis=1) * z, axis=1)


def ring_section(loops: np.ndarray):  # -> SectionMesh
    """
    Structured Quad4 ring between nested closed loops.

    Args:
        loops: (n_layers + 1, m, 2) closed polylines in (y, z), index 0 = outer
               loop, index −1 = inner loop; all loops CCW with matching point j.

    Returns:
        SectionMesh with node id = l · m + j, CCW quads
        [outer_j, outer_{j+1}, inner_{j+1}, inner_j] per layer, outer edges on
        loop 0 and inner edges on the last loop.  The periodic seam shares nodes.
    """
    from fahts.core.heat.section_mesh.section_mesh import SectionMesh

    loops = np.asarray(loops, dtype=float)
    n_loop, m, _ = loops.shape
    nodes = loops.reshape(-1, 2)
    j = np.arange(m)
    j1 = (j + 1) % m
    quads = []
    for lay in range(n_loop - 1):
        o, i = lay * m, (lay + 1) * m
        quads.append(np.stack([o + j, o + j1, i + j1, i + j], axis=1))
    quads_arr = np.concatenate(quads).astype(np.intp)
    last = (n_loop - 1) * m
    return SectionMesh(
        nodes=nodes,
        quads=quads_arr,
        outer_edge_pairs=np.stack([j, j1], axis=1).astype(np.intp),
        inner_edge_pairs=np.stack([last + j1, last + j], axis=1).astype(np.intp),
    )


def rect_loop(y0: float, y1: float, z0: float, z1: float,
              n_y: int, n_z: int) -> np.ndarray:
    """
    (2 (n_y + n_z), 2) CCW points around a rectangle, starting at (y0, z0):
    bottom edge (n_y segments), right edge (n_z), top edge (n_y), left edge (n_z).
    """
    ty = np.arange(n_y) / n_y
    tz = np.arange(n_z) / n_z
    bottom = np.stack([y0 + (y1 - y0) * ty, np.full(n_y, z0)], axis=1)
    right = np.stack([np.full(n_z, y1), z0 + (z1 - z0) * tz], axis=1)
    top = np.stack([y1 - (y1 - y0) * ty, np.full(n_y, z1)], axis=1)
    left = np.stack([np.full(n_z, y0), z1 - (z1 - z0) * tz], axis=1)
    return np.concatenate([bottom, right, top, left])
