"""
Radiation scene of all solid members, and shielding of directional heat sources.

``build_scene`` triangulates every boundary face of every 3-D member (outer, inner and end
faces, global coordinates) into one ``TriangleScene``; each triangle is owned by a global
face id so rays can report which member surface they hit.

Shielding: a face only receives a directional source's flux if the source is visible.
    RadiationBall  — visibility = fraction of unblocked rays to 19 points on the ball's
                     visible front cap (the radiating flame surface, so steel inside the
                     ball does not block it; partial shading); faces inside the ball
                     (engulfed) keep the full flux.
    ConcentratedSource — one ray to the point.
    LineSource     — one ray per sub-source; the unit-power pattern is re-summed with the
                     sub-source visibilities.
Rays start at the face centroid, offset slightly along the outward normal, so a member
shades itself correctly (e.g. an I-beam web behind its flange) but never its own face.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass

import numpy as np

from fahts.core.heat.radiation.raycast import TriangleScene
from fahts.core.heat.solid_mesh import SolidMesh

log = logging.getLogger(__name__)


@dataclass
class MemberFaces:
    """Boundary-face data of one member in the radiation scene."""
    eid: int
    face_offset: int          # global id of the member's first boundary face
    n_faces: int
    outer: np.ndarray         # member-local indices of FACE_OUTER faces (solver order)
    centroids: np.ndarray     # (n_outer, 3) global
    normals: np.ndarray       # (n_outer, 3) global unit outward
    areas: np.ndarray         # (n_outer,)
    corners: np.ndarray       # (n_outer, 4, 3) global


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0.0, n, 1.0)


def build_scene(eids: list[int], meshes: dict, positions: dict
                ) -> tuple[TriangleScene | None, dict[int, MemberFaces]]:
    """Triangulate all boundary faces of all SolidMesh members (global coordinates)."""
    tris: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    members: dict[int, MemberFaces] = {}
    offset = 0
    for eid in eids:
        mesh = meshes.get(eid)
        pos = positions.get(eid)
        if not isinstance(mesh, SolidMesh) or pos is None:
            continue
        P = np.asarray(pos, dtype=float)[np.asarray(mesh.faces)]          # (n_f, 4, 3)
        tris.append(np.concatenate([P[:, [0, 1, 2]], P[:, [0, 2, 3]]]))
        ids = offset + np.arange(len(P))
        owner.append(np.concatenate([ids, ids]))
        outer = mesh.outer_face_indices
        Po = P[outer]
        nrm = _unit(np.cross(Po[:, 2] - Po[:, 0], Po[:, 3] - Po[:, 1]))
        # The beam frame (x, y = x × z, z) is LEFT-handed, so local→global mirrors the
        # mesh and flips cross-product normals inward — detect via the signed hex volume.
        if SolidMesh(np.asarray(pos, dtype=float), mesh.hexes[:1], mesh.faces[:0],
                     mesh.face_group[:0]).hex_volumes()[0] < 0.0:
            nrm = -nrm
        members[eid] = MemberFaces(eid=eid, face_offset=offset, n_faces=len(P), outer=outer,
                                   centroids=Po.mean(axis=1), normals=nrm,
                                   areas=mesh.face_areas()[outer], corners=Po)
        offset += len(P)
    if not tris:
        return None, members
    return TriangleScene(np.concatenate(tris), np.concatenate(owner)), members


def _offset_origins(scene: TriangleScene, cents: np.ndarray, normals: np.ndarray) -> np.ndarray:
    return cents + normals * (1e-6 * scene.scale + 1e-6)


def _disc_samples(n_rings: tuple[int, ...] = (1, 6, 12)) -> np.ndarray:
    """(n, 2) points on the unit disc, roughly equal-area rings (19 points)."""
    pts = [np.zeros(2)]
    radii = (0.5, 0.87)
    for r, n in zip(radii, n_rings[1:]):
        a = 2 * np.pi * (np.arange(n) + 0.5 * (r > 0.6)) / n
        pts.extend(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.array(pts)


def ball_visibility(scene: TriangleScene, cents: np.ndarray, normals: np.ndarray,
                    ball) -> np.ndarray:
    """Fraction of the ball visible from each face (1 inside the ball)."""
    C = np.asarray(ball.center, dtype=float)
    R = float(ball.radius)
    w = C - cents
    d = np.linalg.norm(w, axis=1)
    vis = np.ones(len(cents))
    out = d > R
    if not np.any(out):
        return vis
    wu = _unit(w[out])
    # orthonormal basis ⟂ view direction
    helper = np.where(np.abs(wu[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = _unit(np.cross(wu, helper))
    v = np.cross(wu, u)
    S = _disc_samples()                                                  # (s, 2)
    # targets on the ball's VISIBLE front cap (the radiating flame surface): steel inside
    # the ball must not block its own fire ball
    rho2 = np.minimum((S ** 2).sum(axis=1), 1.0)
    cap = np.sqrt(1.0 - rho2)                                            # (s,)
    targets = (C[None, None, :]
               + R * (S[None, :, 0:1] * u[:, None, :] + S[None, :, 1:2] * v[:, None, :])
               - R * cap[None, :, None] * wu[:, None, :])                # (m, s, 3)
    O0 = _offset_origins(scene, cents[out], normals[out])
    O = np.repeat(O0, len(S), axis=0)
    blocked = scene.blocked(O, targets.reshape(-1, 3)).reshape(-1, len(S))
    # only cap points ABOVE the face's tangent plane count: the part of a large, close
    # ball below the horizon is already excluded by the flux formula's cos θ, and rays
    # to it would just pass through the face's own member
    above = np.einsum("msk,mk->ms", targets - O0[:, None, :], normals[out]) > 0.0
    n_ok = above.sum(axis=1)
    free = (~blocked & above).sum(axis=1)
    vis[out] = np.where(n_ok > 0, free / np.maximum(n_ok, 1), 1.0)
    return vis


def point_visibility(scene: TriangleScene, cents: np.ndarray, normals: np.ndarray,
                     point: np.ndarray) -> np.ndarray:
    """1 where the point is visible from the face, else 0."""
    O = _offset_origins(scene, cents, normals)
    T = np.broadcast_to(np.asarray(point, dtype=float), O.shape)
    return 1.0 - scene.blocked(O, T).astype(float)


def line_source_pattern(scene: TriangleScene, cents: np.ndarray, normals: np.ndarray,
                        src, n_segments: int = 10) -> np.ndarray:
    """Unit-power LineSource flux pattern with per-sub-source shielding."""
    unit = dataclasses.replace(src, power=1.0)
    positions, energies = unit._sub_sources(0.0, n_segments)
    q = np.zeros(len(cents))
    for pos, dE in zip(positions, energies):
        r = cents - pos
        d = np.linalg.norm(r, axis=1)
        ok = d > 1e-9
        ds = np.where(ok, d, 1.0)
        cos_t = np.maximum(-np.einsum("ij,ij->i", normals, r / ds[:, None]), 0.0)
        contrib = np.where(ok, dE / (4.0 * math.pi * ds * ds), 0.0) * cos_t
        lit = contrib > 0.0
        if np.any(lit):
            vis = np.zeros(len(cents))
            vis[lit] = point_visibility(scene, cents[lit], normals[lit], pos)
            q += contrib * vis
    return q


def apply_source_shielding(scene: TriangleScene, members: dict[int, MemberFaces],
                           solvers: dict, balls: dict[int, object]) -> dict[str, float]:
    """
    Scale every solid member's directional flux by source visibility (in place).

    ``balls``: {eid: RadiationBall used for that member (static q_per_face)}.
    Point / line sources are read from the member's TimeVaryingFaceFlux schedule.
    Returns simple statistics for logging.
    """
    from fahts.core.heat.bc.face_flux import TimeVaryingFaceFlux
    from fahts.core.heat.sources.concentrated_source import ConcentratedSource
    from fahts.core.heat.sources.line_source import LineSource

    n_lit = n_shaded = 0.0
    for eid, mf in members.items():
        sv = solvers.get(eid)
        if sv is None or getattr(sv, "_q_per_face", None) is None:
            continue
        sched = sv._q_face_fn if isinstance(sv._q_face_fn, TimeVaryingFaceFlux) else None
        if sched is None:
            ball = balls.get(eid)
            if ball is None:
                continue
            q = np.asarray(sv._q_per_face, dtype=float)
            lit = q > 0.0
            if not np.any(lit):
                continue
            vis = np.ones(len(q))
            vis[lit] = ball_visibility(scene, mf.centroids[lit], mf.normals[lit], ball)
            sv._q_per_face = q * vis
            n_lit += float(lit.sum())
            n_shaded += float((1.0 - vis[lit]).sum())
            continue
        if sched.static is not None and balls.get(eid) is not None:
            lit = sched.static > 0.0
            vis = np.ones(len(sched.static))
            vis[lit] = ball_visibility(scene, mf.centroids[lit], mf.normals[lit], balls[eid])
            sched.static = sched.static * vis
        new_terms = []
        for (power, pattern), src in zip(sched.terms, sched.sources):
            if isinstance(src, LineSource):
                pattern = line_source_pattern(scene, mf.centroids, mf.normals, src)
            elif isinstance(src, ConcentratedSource):
                lit = pattern > 0.0
                vis = np.zeros(len(pattern))
                vis[lit] = point_visibility(scene, mf.centroids[lit], mf.normals[lit],
                                            src.center)
                n_lit += float(lit.sum())
                n_shaded += float((1.0 - vis[lit]).sum())
                pattern = pattern * vis
            new_terms.append((power, pattern))
        sched.terms = new_terms
        sv._q_per_face = sched(0.0)
    return {"lit_faces": n_lit, "shaded_face_equiv": n_shaded}
