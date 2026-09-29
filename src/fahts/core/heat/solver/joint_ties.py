"""
Thermal surface ties between solid members at structural joints (3-D path).

Frame models put members on their centre-lines, so at a joint the solid meshes of
different members overlap or abut without sharing nodes: the end face of a brace
sits inside the chord (often inside its hollow cavity), and a plate edge runs along
the centre-line of its supporting beam.  Merging nodes there would distort hexes;
nearest-node links only reach the few nodes that happen to be close.

Instead every *source* node p (a free beam-end FACE_END node at the joint, or a
plate FACE_END node near its partner) is tied to the CLOSEST POINT q on the
partner member's surface.  q lies on a boundary triangle (v1, v2, v3) with
barycentric weights λ, and the tie is a steel bridge of total conductance

    G_p = k_steel · A_p / max(d_pq, h_a)        [W/K]

split into star links p–v_k with conductance G_p·λ_k (a graph Laplacian with
non-negative weights → M-matrix, no over/undershoot).

(A_p = tributary end-face area of p, h_a = shortest hex edge of the source member,
so a tie is never stiffer than one element layer).  Every row of K_tie sums to zero
(no heat created or destroyed), and K_tie is symmetric positive semi-definite, so K
stays SPD for CG.

Source selection at a beam joint: a beam end whose end-face nodes are already
merged with another member's nodes (collinear continuation) is a *through* end and
does not tie; only free ends tie.  If a joint has no free end (e.g. an X-joint of
two through members) all ends tie.  Plates always act as sources.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from fahts.core.heat.solid_mesh import FACE_END, FACE_OUTER, SolidMesh
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection

log = logging.getLogger(__name__)

#: Reference steel conductivity for ties [W/(m·K)].
K_STEEL_TIE: float = 45.0
#: Candidate triangles examined per source point.
_K_NEAREST_TRI: int = 12


# ── geometry helpers ──────────────────────────────────────────────────────────

def half_diagonal(sec) -> float:
    """Half the bounding-box diagonal of a cross-section [m] (plates: half thickness)."""
    if isinstance(sec, BoxSection):
        return 0.5 * math.hypot(sec.W, sec.H)
    if isinstance(sec, PipeSection):
        return 0.5 * sec.outer_diameter
    if isinstance(sec, ISection):
        return 0.5 * math.hypot(max(sec.bf_top, sec.bf_bot), sec.h)
    if isinstance(sec, PlateSection):
        return 0.5 * float(getattr(sec, "thickness", 0.0))
    return 0.0


def closest_point_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closest point of each triangle (a, b, c) to p — vectorised (Ericson, RTCD §5.1.5).

    All inputs (n, 3).  Returns (q (n, 3), bary (n, 3)) with q = Σ bary_k · vertex_k.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    n = len(p)
    bary = np.zeros((n, 3))
    done = np.zeros(n, dtype=bool)

    def _set(mask: np.ndarray, w: np.ndarray) -> None:
        m = mask & ~done
        bary[m] = w[m]
        done[m] = True

    one = np.ones(n)
    zero = np.zeros(n)
    _set((d1 <= 0) & (d2 <= 0), np.stack([one, zero, zero], 1))            # vertex a
    _set((d3 >= 0) & (d4 <= d3), np.stack([zero, one, zero], 1))           # vertex b
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        _set((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v, v, zero], 1))   # edge ab
        _set((d6 >= 0) & (d5 <= d6), np.stack([zero, zero, one], 1))       # vertex c
        vb = d5 * d2 - d1 * d6
        w = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        _set((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w, zero, w], 1))   # edge ac
        va = d3 * d6 - d5 * d4
        den = (d4 - d3) + (d5 - d6)
        w = np.where(den != 0, (d4 - d3) / den, 0.0)
        _set((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
             np.stack([zero, 1 - w, w], 1))                                 # edge bc
        denom = va + vb + vc
        denom = np.where(denom != 0, denom, 1.0)
        v = vb / denom
        w = vc / denom
        _set(np.ones(n, dtype=bool), np.stack([1 - v - w, v, w], 1))       # interior
    q = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return q, bary


class _Surface:
    """Triangulated target surface of one member (global coords) with a KD-tree."""

    def __init__(self, mesh: SolidMesh, pos: np.ndarray, groups: tuple[int, ...]) -> None:
        f = np.asarray(mesh.faces)[np.isin(mesh.face_group, groups)]
        tri = np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])      # (n_tri, 3) local ids
        self.tri = tri
        self.pts = pos[tri]                                            # (n_tri, 3, 3)
        self.tree = cKDTree(self.pts.mean(axis=1))

    def closest(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For points p (m, 3): distance (m,), triangle local node ids (m, 3), bary (m, 3)."""
        k = min(_K_NEAREST_TRI, len(self.tri))
        _, idx = self.tree.query(p, k=k)
        idx = np.asarray(idx).reshape(len(p), k)
        P = np.repeat(p, k, axis=0)
        T = self.pts[idx.ravel()]
        q, bary = closest_point_on_triangles(P, T[:, 0], T[:, 1], T[:, 2])
        d = np.linalg.norm(P - q, axis=1).reshape(len(p), k)
        best = np.argmin(d, axis=1)
        rows = np.arange(len(p))
        tri_best = idx[rows, best]
        return d[rows, best], self.tri[tri_best], bary.reshape(len(p), k, 3)[rows, best]


def _end_node_area(mesh: SolidMesh) -> np.ndarray:
    """(n_nodes,) tributary FACE_END area per node [m²]."""
    fe = mesh.face_indices(FACE_END)
    a = np.zeros(mesh.n_nodes)
    if len(fe):
        np.add.at(a, np.asarray(mesh.faces)[fe].ravel(), np.repeat(mesh.face_areas()[fe] / 4, 4))
    return a


def _min_edge(mesh: SolidMesh) -> float:
    X = np.asarray(mesh.nodes)[np.asarray(mesh.hexes)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return float(min(np.linalg.norm(X[:, i] - X[:, j], axis=1).min() for i, j in e))


def _dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-30), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


# ── public API ────────────────────────────────────────────────────────────────

def build_joint_ties(
    eids: list[int],
    meshes: dict,
    global_positions: dict[int, np.ndarray | None],
    gdof_map: dict[int, np.ndarray],
    n_global: int,
    model,
    k_steel: float = K_STEEL_TIE,
    tol_min: float = 0.025,
) -> sp.csr_matrix | None:
    """
    Assemble the constant joint-tie conductance matrix (n_global × n_global CSR).

    Returns None when no tie is created (e.g. isolated members).
    """
    elements = model.elements
    shells = getattr(model, "shell_elements", {})
    solid = [e for e in eids if isinstance(meshes.get(e), SolidMesh)
             and global_positions.get(e) is not None and (e in elements or e in shells)]
    if len(solid) < 2:
        return None

    def sec_of(e: int):
        el = elements.get(e) or shells.get(e)
        return model.sections.get(el.geom_id)

    def nodes_of(e: int) -> set[int]:
        return {elements[e].n1, elements[e].n2} if e in elements else set(shells[e].nodes)

    at_node: dict[int, list[int]] = {}
    for e in solid:
        for nid in nodes_of(e):
            at_node.setdefault(nid, []).append(e)

    dof_owner: dict[int, set[int]] = {}           # global dof → members using it
    for e in solid:
        for g in np.unique(gdof_map[e]):
            dof_owner.setdefault(int(g), set()).add(e)

    area = {e: _end_node_area(meshes[e]) for e in solid}
    hmin = {e: _min_edge(meshes[e]) for e in solid}
    surf: dict[int, _Surface] = {}

    def surface(e: int) -> _Surface:
        if e not in surf:
            groups = (FACE_OUTER,) if e in elements else (FACE_OUTER, FACE_END)
            surf[e] = _Surface(meshes[e], np.asarray(global_positions[e], float), groups)
        return surf[e]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    n_ties = 0

    def add_ties(a: int, b: int, cand: np.ndarray, r_max: float) -> None:
        nonlocal n_ties
        if len(cand) == 0:
            return
        g_a = np.asarray(gdof_map[a])
        # skip nodes already sharing a DOF with b (merged)
        cand = np.array([i for i in cand if b not in dof_owner.get(int(g_a[i]), ())],
                        dtype=np.intp)
        if len(cand) == 0:
            return
        p = np.asarray(global_positions[a], float)[cand]
        d, tri, lam = surface(b).closest(p)
        ok = d <= r_max
        if not np.any(ok):
            return
        cand, d, tri, lam = cand[ok], d[ok], tri[ok], lam[ok]
        G = k_steel * area[a][cand] / np.maximum(d, hmin[a])
        # Star links p–v_k with conductance G·λ_k (Σλ = 1 → total G): a graph Laplacian
        # with non-negative weights, i.e. an M-matrix (no positive couplings, unlike the
        # interpolated form G·w·wᵀ whose λ_k·λ_l cross terms are positive).
        gp = g_a[cand]
        gv = np.asarray(gdof_map[b])[tri]                                         # (m, 3)
        gk = G[:, None] * lam                                                     # (m, 3)
        pi = np.repeat(gp, 3)
        vk = gv.ravel()
        wk = gk.ravel()
        rows.append(np.concatenate([pi, vk, pi, vk]))
        cols.append(np.concatenate([pi, vk, vk, pi]))
        vals.append(np.concatenate([wk, wk, -wk, -wk]))
        n_ties += len(cand)

    handled_pairs: set[tuple[int, int]] = set()
    for nid, members in at_node.items():
        if len(members) < 2:
            continue
        tol = max(tol_min, *(2.0 * half_diagonal(sec_of(e)) for e in members if e in shells),
                  0.0)

        # beam ends at this joint: local end-face node ids + "free" flag
        ends: dict[int, np.ndarray] = {}
        free: dict[int, bool] = {}
        for e in members:
            if e not in elements:
                continue
            el = elements[e]
            x_t = 0.0 if el.n1 == nid else el.length
            xl = np.asarray(meshes[e].nodes, float)[:, 0]
            idx = np.flatnonzero((area[e] > 0.0)
                                 & (np.abs(xl - x_t) <= 1e-6 * max(1.0, el.length)))
            ends[e] = idx
            g = np.asarray(gdof_map[e])[idx]
            free[e] = not any(len(dof_owner.get(int(x), ())) > 1 for x in g)
        sources = [e for e in ends if free[e]] or list(ends)

        for a in sources:
            for b in members:
                if b == a:
                    continue
                r = half_diagonal(sec_of(b)) + tol if b in elements else tol
                add_ties(a, b, ends[a], r)

        # plates: edge nodes near each partner sharing this joint (once per pair)
        for a in members:
            if a in elements:
                continue
            fe_nodes = np.flatnonzero(area[a] > 0.0)
            pos_a = np.asarray(global_positions[a], float)
            for b in members:
                if b == a or (a, b) in handled_pairs:
                    continue
                handled_pairs.add((a, b))
                if b in elements:
                    el = elements[b]
                    p1 = np.asarray(model.nodes[el.n1].xyz, float)
                    p2 = np.asarray(model.nodes[el.n2].xyz, float)
                    if el.ecc1 is not None:
                        p1 = p1 + np.asarray(el.ecc1, float)
                    if el.ecc2 is not None:
                        p2 = p2 + np.asarray(el.ecc2, float)
                    r = half_diagonal(sec_of(b)) + tol
                    near = fe_nodes[_dist_to_segment(pos_a[fe_nodes], p1, p2) <= r]
                else:
                    r = tol
                    near = fe_nodes
                add_ties(a, b, near, r)

    if not vals:
        return None
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_global, n_global))
    log.info("3-D joint ties: %d surface ties (trace %.3g W/K).", n_ties, float(K.diagonal().sum()))
    return K
