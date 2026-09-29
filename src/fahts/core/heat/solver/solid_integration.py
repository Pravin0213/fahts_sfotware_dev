"""
Integration helpers for the 3-D Hex8 solid path of ``run_analysis``.

Everything that turns a structural member + heat sources into the inputs of
``SolidTransientSolver`` lives here, plus the 3-D global DOF map and the
joint-link conductance matrix consumed by ``GlobalThermalSolver``:

- ``build_beam_solid_mesh``      BOX / IHPROFIL / PIPE → SolidMesh (beam-local)
- ``outer_face_geometry``        FACE_OUTER corners / centroids / outward normals (global)
- ``zone_face_exposure``         FireZone exposure flag per outer face
- ``face_view_factors``          §3.3.4 double-area view factor, vectorised
- ``ball_face_flux``             RadiationBall incident flux per outer face
- ``solid_M_extra``              enclosed-gas capacity of hollow BOX/PIPE on inner nodes
- ``build_solid_dof_map``        safe proximity node merging across members
- ``build_joint_links``          conductance links between members at structural joints

All per-face arrays are ordered like ``mesh.outer_face_indices``.  Solid faces carry
TRUE outward normals on every side, so no ``double_sided`` / ``n_exposed_sides``
corrections are needed (or allowed) here.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from fahts.core.heat.solid_mesh import (
    BoxSolidMesher,
    IProfileSolidMesher,
    PipeSolidMesher,
    SolidMesh,
)
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection

log = logging.getLogger(__name__)

#: Reference steel conductivity used for joint links [W/(m·K)].
K_STEEL_LINK: float = 45.0

_HEX_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.intp)


# ── Mesh construction ─────────────────────────────────────────────────────────

def axial_divisions(sec, length: float, config, n_min: int) -> int:
    """
    Axial element count for a 3-D member: at least *n_min*, and enough that the axial
    element length ≤ ``config.axial_aspect_3d`` × the cross-section element size.
    """
    aspect = float(getattr(config, "axial_aspect_3d", 0.0) or 0.0)
    if aspect <= 0.0:
        return int(n_min)
    if isinstance(sec, BoxSection):
        h_cs = max(sec.W / config.n_top, sec.H / config.n_side)
    elif isinstance(sec, ISection):
        h_cs = max(max(sec.bf_top, sec.bf_bot) / (config.n_top_i + 1), sec.h / config.n_side_i)
    elif isinstance(sec, PipeSection):
        c_circ = getattr(config, "c_circ_3d", max(config.c_circ, 12))
        h_cs = math.pi * sec.outer_diameter / c_circ
    else:
        return int(n_min)
    return max(int(n_min), int(math.ceil(length / (aspect * h_cs))))


def build_beam_solid_mesh(sec, length: float, config) -> SolidMesh:
    """Build the Hex8 solid mesh (beam-local coords) for a BOX / IHPROFIL / PIPE member.

    Axial divisions follow ``axial_divisions`` (configured counts are minimums)."""
    nl = int(config.n_layers_3d)
    if isinstance(sec, BoxSection):
        return BoxSolidMesher(sec, length, n_top=config.n_top, n_side=config.n_side,
                              n_length=axial_divisions(sec, length, config, config.n_length),
                              n_layers=nl).build()
    if isinstance(sec, ISection):
        return IProfileSolidMesher(sec, length, n_top=config.n_top_i, n_side=config.n_side_i,
                                   n_bottom=config.n_bottom_i,
                                   n_length=axial_divisions(sec, length, config,
                                                            config.n_length_i),
                                   n_layers=nl).build()
    if isinstance(sec, PipeSection):
        c_circ = getattr(config, "c_circ_3d", max(config.c_circ, 12))
        return PipeSolidMesher(sec, length, c_circ=c_circ,
                               n_length=axial_divisions(sec, length, config, config.n_length_p),
                               n_layers=nl).build()
    raise TypeError(f"Unsupported section type for 3-D solid mesh: {type(sec).__name__}")


def min_edge_length(mesh: SolidMesh) -> float:
    """Shortest Hex8 edge of *mesh* [m]."""
    if mesh.n_hexes == 0:
        return float("inf")
    p = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.hexes)[:, _HEX_EDGES]]
    return float(np.linalg.norm(p[..., 1, :] - p[..., 0, :], axis=-1).min())


def section_wall_thickness(sec) -> float:
    """Largest wall / plate thickness of a section [m] (0 if unknown)."""
    if isinstance(sec, BoxSection):
        return float(max(sec.T_side, sec.T_top, sec.T_bot))
    if isinstance(sec, PipeSection):
        return float(sec.thickness)
    if isinstance(sec, ISection):
        return float(max(sec.tw, sec.tf_top, sec.tf_bot))
    if isinstance(sec, PlateSection):
        return float(sec.thickness)
    return 0.0


# ── Face geometry & boundary data ─────────────────────────────────────────────

def outer_face_geometry(
    mesh: SolidMesh,
    R: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global geometry of the FACE_OUTER faces.

    Args:
        mesh:   SolidMesh (beam-local if R given, else already global).
        R:      (3, 3) local→global rotation (columns = local x, y, z) or None.
        origin: (3,) global position of the local origin (n1 + ecc1) or None.

    Returns:
        corners (n_o, 4, 3), centroids (n_o, 3), unit outward normals (n_o, 3).
    """
    idx = mesh.outer_face_indices
    corners = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.faces)[idx]]
    normals = mesh.face_normals()[idx]
    if R is not None:
        corners = corners @ R.T
        normals = normals @ R.T
    if origin is not None:
        corners = corners + np.asarray(origin, dtype=float)
    centroids = corners.mean(axis=1)
    return corners, centroids, normals


def zone_face_exposure(centroids: np.ndarray, zones: Iterable) -> np.ndarray:
    """(n,) 1.0 where the face centroid lies inside any active FireZone, else 0.0."""
    pts = np.asarray(centroids, dtype=float)
    flags = np.zeros(len(pts), dtype=bool)
    for z in zones:
        if not getattr(z, "active", True):
            continue
        c = np.asarray(z.center, dtype=float)
        half = np.asarray(z.dims, dtype=float) / 2.0
        flags |= np.all(np.abs(pts - c) <= half, axis=1)
    return flags.astype(float)


def face_view_factors(
    corners: np.ndarray,
    normals: np.ndarray,
    zones: Iterable,
    n_sub: int = 4,
    n_steel_sub: int = 2,
) -> np.ndarray:
    """
    FAHTS §3.3.4 double-area view factor for every face (vectorised).

    Identical formula to ``geometric_view_factor_double_area`` (no shadowing):
        F = (1/A₁) Σ_i Σ_j cosθ_i cosθ_j / (π r²) · A_i A_j,   F ≤ 1
    with the steel face split into n_steel_sub² bilinear sub-patches and the fire
    zone faces into ``FireZone.face_patches(n_sub)``.
    """
    patches = [p for z in zones for p in z.face_patches(n_sub)]
    P = np.asarray(corners, dtype=float)
    n_f = len(P)
    if not patches or n_f == 0:
        return np.zeros(n_f)
    pc = np.array([p[0] for p in patches], dtype=float)            # (m, 3)
    pn = np.array([p[1] for p in patches], dtype=float)            # (m, 3)
    pa = np.array([p[2] for p in patches], dtype=float)            # (m,)

    A_tot = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1)
    ns = n_steel_sub
    s = (np.arange(ns) + 0.5) / ns
    S, T = np.meshgrid(s, s, indexing="ij")
    S, T = S.ravel(), T.ravel()
    w = np.stack([(1 - S) * (1 - T), S * (1 - T), S * T, (1 - S) * T], axis=1)  # (k, 4)
    n_hat = np.asarray(normals, dtype=float)

    F = np.zeros(n_f)
    chunk = max(1, 200_000 // max(1, len(patches) * len(w)))
    for a in range(0, n_f, chunk):
        b = min(n_f, a + chunk)
        C = np.einsum("kc,fcx->fkx", w, P[a:b])                    # (f, k, 3)
        r = pc[None, None] - C[:, :, None]                         # (f, k, m, 3)
        d = np.linalg.norm(r, axis=-1)
        ok = d >= 1e-9
        d_s = np.where(ok, d, 1.0)
        cos_i = np.einsum("fx,fkmx->fkm", n_hat[a:b], r) / d_s
        cos_j = -np.einsum("mx,fkmx->fkm", pn, r) / d_s
        contrib = np.where(ok & (cos_i > 0.0) & (cos_j > 0.0),
                           cos_i * cos_j / (math.pi * d_s * d_s) * pa[None, None], 0.0)
        sub_a = A_tot[a:b] / (ns * ns)
        F[a:b] = contrib.sum(axis=(1, 2)) * sub_a
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(A_tot > 1e-18, F / np.where(A_tot > 0.0, A_tot, 1.0), 0.0)
    return np.minimum(F, 1.0)


def ball_face_flux(ball, centroids: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    RadiationBall incident flux per face (vectorised ``RadiationBall.incident_flux``):
    q = flux if d ≤ radius, else flux·(radius/d)²·max(cosθ, 0).
    """
    c = np.asarray(ball.center, dtype=float)
    r_vec = c - np.asarray(centroids, dtype=float)
    d = np.linalg.norm(r_vec, axis=1)
    d_s = np.where(d > 0.0, d, 1.0)
    cos_t = np.einsum("ij,ij->i", np.asarray(normals, dtype=float), r_vec) / d_s
    q_ext = ball.flux * (ball.radius / d_s) ** 2 * np.maximum(cos_t, 0.0)
    return np.where(d <= ball.radius, float(ball.flux), q_ext)


def solid_M_extra(sec, mesh: SolidMesh, length: float, rho_c: float) -> np.ndarray | None:
    """
    Enclosed-gas heat capacity (§3.3.5) of hollow BOX / PIPE members [J/K per node].

    Total capacity A_cavity · L · ρc_gas is distributed over the FACE_INNER nodes in
    proportion to their tributary inner-surface area.
    """
    if isinstance(sec, BoxSection):
        A_inner = sec.inner_height * sec.inner_width
    elif isinstance(sec, PipeSection):
        A_inner = math.pi * sec.inner_radius ** 2
    else:
        return None
    if A_inner <= 0.0 or rho_c <= 0.0:
        return None
    fi = mesh.inner_face_indices
    if len(fi) == 0:
        return None
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, np.asarray(mesh.faces)[fi].ravel(), np.repeat(mesh.face_areas()[fi] / 4.0, 4))
    if w.sum() <= 0.0:
        return None
    return A_inner * length * rho_c * w / w.sum()


# ── Global DOF map for solid meshes ───────────────────────────────────────────

def build_solid_dof_map(
    eids: list[int],
    meshes: dict,
    global_positions: dict[int, np.ndarray | None],
    tol: float = 1e-3,
) -> tuple[dict[int, np.ndarray], int]:
    """
    Merge co-located nodes of DIFFERENT members into shared global DOFs (3-D path).

    Safeguards against collapsing a member's own hexes:
      * per-member tolerance ``tol_e = min(tol, 0.2 · shortest hex edge)`` — a pair is
        merged only if its distance ≤ min(tol_a, tol_b);
      * nodes of the same member are never paired;
      * union-find tracks the set of members in each merged group and REJECTS any
        union that would put two nodes of the same member into one DOF (prevents
        chain merges A₁–B–A₂).  Pairs are processed closest-first.

    Members without spatial positions (TRISHELL 1-D) get independent DOFs appended.
    """
    lab_eid: list[np.ndarray] = []
    lab_loc: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    tol_of: dict[int, float] = {}
    order = {eid: k for k, eid in enumerate(eids)}
    for eid in eids:
        pos = global_positions.get(eid)
        if pos is None:
            continue
        m = meshes[eid]
        tol_of[eid] = min(tol, 0.2 * min_edge_length(m)) if isinstance(m, SolidMesh) else tol
        coords.append(np.asarray(pos, dtype=float))
        lab_eid.append(np.full(len(pos), order[eid], dtype=np.intp))
        lab_loc.append(np.arange(len(pos), dtype=np.intp))

    n_sp = int(sum(len(c) for c in coords))
    parent = np.arange(n_sp)
    rejected = 0
    if n_sp:
        X = np.vstack(coords)
        owner = np.concatenate(lab_eid)
        tol_by_owner = np.full(len(eids), tol)
        for eid, tv in tol_of.items():
            tol_by_owner[order[eid]] = tv
        pairs = cKDTree(X).query_pairs(tol, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.linalg.norm(X[i] - X[j], axis=1)
            keep = (owner[i] != owner[j]) & (
                d <= np.minimum(tol_by_owner[owner[i]], tol_by_owner[owner[j]])
            )
            i, j, d = i[keep], j[keep], d[keep]
            srt = np.argsort(d, kind="stable")
            members: dict[int, set[int]] = {}

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in zip(i[srt].tolist(), j[srt].tolist()):
                ra, rb = _find(a), _find(b)
                if ra == rb:
                    continue
                ma = members.get(ra, {int(owner[ra])})
                mb = members.get(rb, {int(owner[rb])})
                if ma & mb:
                    rejected += 1
                    continue
                if len(ma) < len(mb):
                    ra, rb, ma, mb = rb, ra, mb, ma
                parent[rb] = ra
                members[ra] = ma | mb
                members.pop(rb, None)
            # full path compression
            for x in range(n_sp):
                parent[x] = _find(x)
    if rejected:
        log.info("3-D node merging: rejected %d unions that would collapse a member.",
                 rejected)

    roots, dof_of_spatial = np.unique(parent, return_inverse=True)
    dof_idx = len(roots)
    gdof_map: dict[int, np.ndarray] = {}
    offset = 0
    for eid in eids:
        pos = global_positions.get(eid)
        n_nodes = meshes[eid].n_nodes
        if pos is None:
            gdof_map[eid] = np.arange(dof_idx, dof_idx + n_nodes, dtype=np.intp)
            dof_idx += n_nodes
        else:
            gdof_map[eid] = dof_of_spatial[offset:offset + n_nodes].astype(np.intp)
            offset += n_nodes
    return gdof_map, dof_idx


def build_joint_links(
    eids: list[int],
    meshes: dict,
    global_positions: dict[int, np.ndarray | None],
    gdof_map: dict[int, np.ndarray],
    n_global: int,
    model,
    k_link: float = K_STEEL_LINK,
    struct_tol_min: float = 0.025,
) -> sp.csr_matrix | None:
    """
    Constant joint conductance matrix between members meeting at structural nodes.

    Delegates to ``joint_ties.build_joint_ties``: free beam-end nodes and plate edge
    nodes are tied to the closest point on the partner member's surface (shape-function
    interpolated), G = k_steel · A_trib / max(d, h_min).  Symmetric, zero row sums
    (energy conserving) and positive semi-definite.  See that module for the physics.
    """
    from fahts.core.heat.solver.joint_ties import build_joint_ties

    return build_joint_ties(eids, meshes, global_positions, gdof_map, n_global, model,
                            k_steel=k_link, tol_min=struct_tol_min)
