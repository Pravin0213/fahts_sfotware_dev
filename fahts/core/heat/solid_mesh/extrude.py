"""
Extrusion of a 2-D cross-section mesh (y-z plane) into a Hex8 solid mesh.

The section is swept along the beam-local x-axis over [0, L] in `n_length`
equal segments.  A CCW cross-section quad (a, b, c, d) over the segment
[x_k, x_{k+1}] becomes the Hex8

    [a_k, b_k, c_k, d_k, a_{k+1}, b_{k+1}, c_{k+1}, d_{k+1}]

whose Jacobian is positive (in-plane CCW × +x).  Clockwise section quads are
flipped automatically.  Node numbering: 3-D node = k · n_2d + i (k = axial
station, i = 2-D node index).

Boundary face groups:
    - cross-section end caps (x = 0, x = L)         → FACE_END
    - lateral faces on `inner_edge_pairs` edges      → FACE_INNER
    - all other lateral boundary faces               → FACE_OUTER
      (edges in neither list are logged and treated as FACE_OUTER)
"""
from __future__ import annotations

import logging

import numpy as np

from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from ._hex_topology import (
    LOCAL_FACE_ZETA_MINUS, LOCAL_FACE_ZETA_PLUS, boundary_faces, quad_signed_areas_2d,
)
from .solid_mesh import FACE_END, FACE_INNER, FACE_OUTER, SolidMesh

logger = logging.getLogger(__name__)


def extrude_section(
    section_mesh: SectionMesh,
    length: float,
    n_length: int,
    section_kind: str = "EXTRUDED",
) -> SolidMesh:
    """
    Sweep a 2-D cross-section mesh along x ∈ [0, length] into a Hex8 SolidMesh.

    Args:
        section_mesh: 2-D Quad4 mesh in (y, z) with outer/inner edge pairs.
        length:       Member length [m] (> 0).
        n_length:     Number of axial element layers (≥ 1).
        section_kind: Informational tag stored on the SolidMesh.

    Returns:
        Validated SolidMesh in beam-local coordinates.
    """
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    if n_length < 1:
        raise ValueError(f"n_length must be >= 1, got {n_length}")

    nodes2 = np.asarray(section_mesh.nodes, dtype=float)
    quads = np.asarray(section_mesh.quads, dtype=np.intp).copy()
    n2 = len(nodes2)

    area = quad_signed_areas_2d(nodes2, quads)
    if np.any(area == 0.0):
        raise ValueError("Degenerate (zero-area) quad in cross-section mesh")
    cw = area < 0.0
    quads[cw] = quads[cw][:, ::-1]

    xs = np.linspace(0.0, length, n_length + 1)
    nodes = np.empty(((n_length + 1) * n2, 3))
    nodes[:, 0] = np.repeat(xs, n2)
    nodes[:, 1] = np.tile(nodes2[:, 0], n_length + 1)
    nodes[:, 2] = np.tile(nodes2[:, 1], n_length + 1)

    k = np.arange(n_length)[:, None, None] * n2                     # (n_len, 1, 1)
    base = quads[None, :, :] + k                                     # (n_len, n_q, 4)
    hexes = np.concatenate([base, base + n2], axis=2).reshape(-1, 8)

    faces, _, local_face = boundary_faces(hexes)
    group = np.full(len(faces), FACE_OUTER, dtype=np.intp)
    is_cap = (local_face == LOCAL_FACE_ZETA_MINUS) | (local_face == LOCAL_FACE_ZETA_PLUS)
    group[is_cap] = FACE_END

    lateral = np.flatnonzero(~is_cap)
    if len(lateral):
        # Lateral hex faces (η/ξ = ±1) contain the 2-D edge as their first two nodes
        edges = np.sort(faces[lateral][:, :2] % n2, axis=1)
        inner_keys = _edge_keys(section_mesh.inner_edge_pairs, n2)
        outer_keys = _edge_keys(section_mesh.outer_edge_pairs, n2)
        ekey = edges[:, 0] * n2 + edges[:, 1]
        is_inner = np.isin(ekey, inner_keys)
        group[lateral[is_inner]] = FACE_INNER
        unknown = ~is_inner & ~np.isin(ekey, outer_keys)
        if np.any(unknown):
            logger.warning(
                "extrude_section: %d boundary edge(s) not in outer/inner lists; "
                "treated as FACE_OUTER", int(np.count_nonzero(unknown)) // n_length,
            )

    mesh = SolidMesh(nodes=nodes, hexes=hexes, faces=faces, face_group=group,
                     section_kind=section_kind)
    mesh.validate()
    return mesh


def _edge_keys(pairs: np.ndarray, n2: int) -> np.ndarray:
    """Unordered-edge integer keys min·n2 + max."""
    p = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(0, dtype=np.intp)
    s = np.sort(p, axis=1)
    return s[:, 0] * n2 + s[:, 1]
