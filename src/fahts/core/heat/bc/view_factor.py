"""
Exposure logic and geometric view factor calculations.

Phase 2 exposure: binary in/out check on beam midpoint.
Phase 3 exposure: per-quad (heat-transfer-element-level) exposure flags —
  each quad centroid is tested independently against fire zones.
Phase 3E+ radiation: FAHTS §3.2.4 net radiation with §3.3.4 geometric view factors.
Shadow detection: ray-based obstruction testing for radiative exchange.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fahts.core.heat.bc.net_flux import net_heat_flux
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.model.element import BeamElement

if TYPE_CHECKING:
    from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh


@dataclass
class ElementHeatBC:
    """Per-element heat-flux boundary conditions at a single time step."""

    eid: int
    face_fluxes: dict[str, float]  # {'top': q, 'bot': q, 'left': q, 'right': q} [W/m²]
    t: float                        # time [s]

    @property
    def total_flux(self) -> float:
        """Sum of all face fluxes [W/m²]."""
        return sum(self.face_fluxes.values())

    @property
    def is_exposed(self) -> bool:
        return any(q > 0.0 for q in self.face_fluxes.values())


# Face labels used throughout (BOX convention: top/bot in local-z, left/right in local-y)
FACE_NAMES: tuple[str, ...] = ("top", "bot", "left", "right")


def exposure_flags(
    beam: BeamElement,
    fire_zones: list[FireZone],
    nodes: dict,
    tolerance: float = 0.0,
) -> dict[str, float]:
    """
    Return per-face exposure fractions for one beam element.

    Checks the beam midpoint and both endpoints so that beams straddling a
    zone boundary are correctly identified.  Within each face, per-quad
    granularity is handled by ``element_quad_exposure_flags``.

    Parameters
    ----------
    beam       : the beam element to test
    fire_zones : list of FireZone objects to test against
    nodes      : FEMModel.nodes dict  {nid: Node}
    tolerance  : extra margin [m] added to each zone's bounding box

    Returns
    -------
    dict[str, float]
        Keys: 'top', 'bot', 'left', 'right'.  Values in [0, 1].
    """
    check_pts = [
        beam.midpoint(nodes),
        nodes[beam.n1].xyz,
        nodes[beam.n2].xyz,
    ]
    for pt in check_pts:
        for zone in fire_zones:
            if zone.active and zone.contains_point(pt, tolerance=tolerance):
                return {face: 1.0 for face in FACE_NAMES}
    return {face: 0.0 for face in FACE_NAMES}


def exposed_element_ids(
    elements: dict[int, BeamElement],
    fire_zones: list[FireZone],
    nodes: dict,
    tolerance: float = 0.0,
) -> set[int]:
    """
    Return the set of element IDs exposed to any active fire zone.

    Checks the beam midpoint and both end-node positions so that elements
    straddling a zone boundary are included in the analysis.

    Parameters
    ----------
    elements   : FEMModel.elements dict
    fire_zones : list of FireZone objects
    nodes      : FEMModel.nodes dict
    tolerance  : extra margin [m] added to each zone bounding box
    """
    active_zones = [z for z in fire_zones if z.active]
    if not active_zones:
        return set()

    result: set[int] = set()
    for beam in elements.values():
        check_pts = [
            beam.midpoint(nodes),
            nodes[beam.n1].xyz,
            nodes[beam.n2].xyz,
        ]
        for pt in check_pts:
            for zone in active_zones:
                if zone.contains_point(pt, tolerance=tolerance):
                    result.add(beam.eid)
                    break
            if beam.eid in result:
                break
    return result


def compute_element_bc(
    beam: BeamElement,
    fire_zones: list[FireZone],
    nodes: dict,
    t: float,
    T_steel: float = 20.0,
    epsilon_steel: float = 0.7,
    tolerance: float = 0.0,
) -> ElementHeatBC | None:
    """
    Compute net heat-flux BCs for one beam element at time *t*.

    Returns None if the beam is not exposed to any fire zone.

    Parameters
    ----------
    beam          : beam element
    fire_zones    : list of FireZone objects
    nodes         : FEMModel.nodes dict
    t             : current time [s]
    T_steel       : current steel surface temperature [°C] (uniform in Phase 2)
    epsilon_steel : steel surface emissivity (EN 1993-1-2 §3.1: 0.7 default)
    tolerance     : zone bounding-box margin [m]
    """
    flags = exposure_flags(beam, fire_zones, nodes, tolerance=tolerance)
    if not any(v > 0.0 for v in flags.values()):
        return None

    midpoint = beam.midpoint(nodes)
    covering_zones = [
        z for z in fire_zones
        if z.active and z.contains_midpoint(midpoint, tolerance=tolerance)
    ]

    # Use the hottest covering zone at time t
    hottest = max(covering_zones, key=lambda z: z.temperature(t))
    T_fire = hottest.temperature(t)
    epsilon_m = hottest.effective_epsilon_fire * epsilon_steel
    h_conv = hottest.h_conv

    face_fluxes: dict[str, float] = {}
    for face in FACE_NAMES:
        fraction = flags[face]
        if fraction > 0.0:
            q = net_heat_flux(T_fire, T_steel, epsilon_m, h_conv)
            face_fluxes[face] = q * fraction
        else:
            face_fluxes[face] = 0.0

    return ElementHeatBC(eid=beam.eid, face_fluxes=face_fluxes, t=t)


def compute_all_bcs(
    elements: dict[int, BeamElement],
    fire_zones: list[FireZone],
    nodes: dict,
    t: float,
    T_steel: float = 20.0,
    epsilon_steel: float = 0.7,
    tolerance: float = 0.0,
) -> list[ElementHeatBC]:
    """
    Compute heat-flux BCs for all exposed elements at time *t*.

    Returns a list containing only the elements that are actually exposed
    (i.e. have at least one non-zero face flux).
    """
    result: list[ElementHeatBC] = []
    for beam in elements.values():
        bc = compute_element_bc(
            beam, fire_zones, nodes, t,
            T_steel=T_steel,
            epsilon_steel=epsilon_steel,
            tolerance=tolerance,
        )
        if bc is not None:
            result.append(bc)
    return result


# ── Per-quad heat-transfer-element-level exposure ────────────────────────────

def _beam_local_frame(
    elem: BeamElement,
    nodes: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (origin, R) for the beam-local → global coordinate transform.

    origin : global position of the beam start node (n1) [m]
    R      : (3, 3) rotation matrix; columns = [local_x, local_y, local_z]
    """
    local_x = np.asarray(elem.direction, dtype=float)
    local_z = np.asarray(elem.local_z, dtype=float)
    local_y = np.cross(local_x, local_z)
    norm_y = np.linalg.norm(local_y)
    local_y = local_y / norm_y if norm_y > 1e-9 else np.array([0.0, 1.0, 0.0])
    R = np.column_stack([local_x, local_y, local_z])
    return nodes[elem.n1].xyz, R


def element_quad_exposure_flags(
    mesh: "BeamSurfaceMesh",
    elem: BeamElement,
    nodes: dict,
    fire_zones: list[FireZone],
    tolerance: float = 0.0,
) -> np.ndarray:
    """
    Per-quad exposure flags (0.0 or 1.0) for a beam element against FireZones.

    Each quad centroid is transformed from beam-local to global coordinates and
    tested against all active fire zones.  This is the heat-transfer-element-level
    exposure described in the FAHTS theory manual: a structural element that
    partially overlaps a zone boundary will have exposed quads near the zone and
    unexposed quads outside it.

    Parameters
    ----------
    mesh       : BeamSurfaceMesh in beam-local coordinates
    elem       : BeamElement (provides direction, local_z, n1 for frame)
    nodes      : FEMModel.nodes dict  {nid: Node}
    fire_zones : list of FireZone objects to test against
    tolerance  : extra margin [m] added to each zone's bounding box

    Returns
    -------
    (n_quads,) float array — 1.0 if centroid is inside any active zone, else 0.0
    """
    active_zones = [z for z in fire_zones if z.active]
    if not active_zones:
        return np.zeros(mesh.n_quads)

    origin, R = _beam_local_frame(elem, nodes)
    flags = np.zeros(mesh.n_quads)

    for q in range(mesh.n_quads):
        centroid_local = mesh.nodes[mesh.quads[q]].mean(axis=0)
        centroid_global = origin + R @ centroid_local
        for zone in active_zones:
            if zone.contains_point(centroid_global, tolerance=tolerance):
                flags[q] = 1.0
                break

    return flags


# ── §3.3.4 Geometric view factor ─────────────────────────────────────────────

def geometric_view_factor(
    patch_centroid: np.ndarray,
    patch_normal: np.ndarray,
    zone_patches: list[tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """
    FAHTS §3.3.4 simplified view factor (area-to-point form).

    Treats the steel patch as a single point (its centroid). Valid when the
    sub-patch area is small relative to the separation distance r.

        F = Σ_j  cosθ_i · cosθ_j / (π · r²) · A_j

    where
      θ_i : angle between the steel patch outward normal and the direction to j
      θ_j : angle between the fire-zone patch inward normal and the direction to i
      r   : distance between patch centroids [m]
      A_j : area of fire-zone sub-patch [m²]

    Only pairs where both cosines are positive contribute (mutual visibility).

    See ``geometric_view_factor_double_area`` for the full §3.3.4 double
    surface integral used in the solver.

    Parameters
    ----------
    patch_centroid : (3,) global centroid of the steel surface patch [m]
    patch_normal   : (3,) outward unit normal of the steel surface patch
    zone_patches   : list of (centroid, inward_normal, area) from
                     FireZone.face_patches()

    Returns
    -------
    float
        Geometric view factor in [0, 1].
    """
    F = 0.0
    for j_centroid, j_normal, A_j in zone_patches:
        r_vec = j_centroid - patch_centroid
        r = float(np.linalg.norm(r_vec))
        if r < 1e-9:
            continue
        r_hat = r_vec / r
        cos_i = float(np.dot(patch_normal, r_hat))   # steel normal vs direction to fire
        cos_j = float(np.dot(j_normal, -r_hat))      # fire inward normal vs direction to steel
        if cos_i <= 0.0 or cos_j <= 0.0:
            continue
        F += cos_i * cos_j / (math.pi * r * r) * A_j
    return min(F, 1.0)


# ── Shadow / obstruction detection ───────────────────────────────────────────

def ray_segment_intersects_beam(
    ray_origin: np.ndarray,
    ray_target: np.ndarray,
    beam_start: np.ndarray,
    beam_end: np.ndarray,
    radius: float,
) -> bool:
    """
    Test whether a ray from *ray_origin* toward *ray_target* is blocked by a
    capsule (swept sphere / cylinder with hemispherical caps) centred on the
    line segment [beam_start, beam_end] with the given *radius*.

    The ray is treated as a finite segment: only intersections between the
    origin and the target are considered.  Intersections at or beyond the
    target are ignored (the target itself is the fire patch; we do not want to
    flag the patch as blocking its own contribution).

    Algorithm: closest-point-between-two-line-segments approach.

    Parameters
    ----------
    ray_origin : (3,) start of the ray (quad centroid in global coords)
    ray_target : (3,) end of the ray (fire-patch centroid in global coords)
    beam_start : (3,) first end-node of the potentially blocking beam
    beam_end   : (3,) second end-node of the potentially blocking beam
    radius     : half-width of the beam capsule [m]

    Returns
    -------
    bool
        True if the capsule blocks the ray (closest approach distance ≤ radius
        and the closest point lies strictly between origin and target).
    """
    d = ray_target - ray_origin          # ray direction vector
    ray_len = float(np.linalg.norm(d))
    if ray_len < 1e-12:
        return False

    seg = beam_end - beam_start          # beam segment direction vector
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-12:
        # Degenerate beam (single point) — treat as sphere test
        diff = beam_start - ray_origin
        t = float(np.dot(diff, d)) / (ray_len * ray_len)
        t = max(0.0, min(t, 1.0))
        closest = ray_origin + t * d
        dist = float(np.linalg.norm(closest - beam_start))
        return dist <= radius and 0.0 < t < 1.0

    # Closest approach between two line segments:
    #   ray   : P(s) = ray_origin + s*d          s ∈ [0, 1]
    #   beam  : Q(t) = beam_start + t*seg        t ∈ [0, 1]
    # Minimise |P(s) - Q(t)|²
    w0 = ray_origin - beam_start
    a = float(np.dot(d, d))        # |d|²
    b = float(np.dot(d, seg))      # d · seg
    c = float(np.dot(seg, seg))    # |seg|²
    e = float(np.dot(d, w0))
    f = float(np.dot(seg, w0))

    denom = a * c - b * b
    if abs(denom) < 1e-18:
        # Parallel lines — use midpoint of segment against ray
        s_c = 0.0
        t_c = max(0.0, min(f / c, 1.0)) if c > 1e-18 else 0.0
    else:
        s_c = (b * f - c * e) / denom
        t_c = (a * f - b * e) / denom
        t_c = max(0.0, min(t_c, 1.0))
        s_c = (b * t_c + e) / a if a > 1e-18 else 0.0

    # s_c must be in (0, 1) exclusive: 0 → at the quad surface (ignore),
    # 1 → at the fire patch (ignore — the patch is not blocking itself).
    _EPS = 1e-6
    if s_c <= _EPS or s_c >= 1.0 - _EPS:
        return False

    s_c = max(0.0, min(s_c, 1.0))
    closest_on_ray  = ray_origin + s_c * d
    closest_on_beam = beam_start + t_c * seg
    dist = float(np.linalg.norm(closest_on_ray - closest_on_beam))
    return dist <= radius


def compute_shadow_mask(
    quad_centroids: np.ndarray,
    fire_patch_centroids: np.ndarray,
    beam_starts: np.ndarray,
    beam_ends: np.ndarray,
    beam_radii: np.ndarray,
) -> np.ndarray:
    """
    Compute a boolean visibility mask for (quad → fire-patch) pairs.

    For each pair (i, j), casts a ray from quad_centroids[i] toward
    fire_patch_centroids[j] and checks whether any of the provided beams
    blocks it.  A beam blocks a ray if the ray passes within *radius* of the
    beam's axis before reaching the fire patch (capsule intersection test).

    Parameters
    ----------
    quad_centroids       : (n_quads, 3) global centroid of each steel sub-patch
    fire_patch_centroids : (n_patches, 3) global centroid of each fire patch
    beam_starts          : (n_beams, 3) first end-node of each blocking beam
    beam_ends            : (n_beams, 3) second end-node of each blocking beam
    beam_radii           : (n_beams,) half-width radius of each blocking beam [m]

    Returns
    -------
    (n_quads, n_patches) bool array
        True  → the pair is *visible* (ray is NOT blocked).
        False → the pair is *shadowed* (ray IS blocked by at least one beam).
    """
    n_q = len(quad_centroids)
    n_p = len(fire_patch_centroids)
    n_b = len(beam_starts)

    visible = np.ones((n_q, n_p), dtype=bool)

    if n_b == 0:
        return visible

    for i in range(n_q):
        origin = quad_centroids[i]
        for j in range(n_p):
            target = fire_patch_centroids[j]
            for k in range(n_b):
                if ray_segment_intersects_beam(
                    origin, target,
                    beam_starts[k], beam_ends[k],
                    float(beam_radii[k]),
                ):
                    visible[i, j] = False
                    break   # one blocking beam is enough

    return visible


def geometric_view_factor_double_area(
    quad_corners: np.ndarray,
    quad_normal: np.ndarray,
    zone_patches: list[tuple[np.ndarray, np.ndarray, float]],
    n_steel_sub: int = 2,
    shadow_beams: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> float:
    """
    FAHTS §3.3.4 full double-area view factor with optional shadow detection.

    Implements the numerical double-area integration:

        F_12 = (1/A1) · Σ_i Σ_j  cosθ_i · cosθ_j / (π · r_ij²) · Ai · Aj

    where surface 1 (the steel quad) is subdivided into n_steel_sub²
    sub-patches and surface 2 (fire zone faces) into the provided zone_patches.
    This is more accurate than the simplified centroid form when the steel quad
    size is comparable to the separation distance.

    When n_steel_sub=1 the result equals geometric_view_factor applied at the
    quad centroid (the simplified form is a special case).

    When *shadow_beams* is provided, a ray is cast from each steel sub-patch
    centroid toward each fire-zone patch centroid.  If the ray is blocked by
    any beam capsule (as tested by ``ray_segment_intersects_beam``), that
    (steel sub-patch → fire patch) pair contributes zero to the view factor.
    Pairs not blocked by any beam are integrated normally.  Passing
    ``shadow_beams=None`` (the default) skips obstruction testing entirely,
    preserving backward compatibility.

    Parameters
    ----------
    quad_corners  : (4, 3) global positions of the steel quad corners (CCW from outside)
    quad_normal   : (3,) outward unit normal of the steel quad
    zone_patches  : list of (centroid, inward_normal, area) from FireZone.face_patches()
    n_steel_sub   : sub-divisions per edge of the steel quad (default 2)
    shadow_beams  : optional list of (start, end, radius) tuples — one entry per
                    potentially blocking beam element.  ``start`` and ``end`` are
                    (3,) global node positions [m]; ``radius`` is the beam's
                    bounding half-width [m].  Default None (no shadow testing).

    Returns
    -------
    float
        Geometric view factor in [0, 1].
    """
    if not zone_patches:
        return 0.0

    P = np.asarray(quad_corners, dtype=float)   # (4, 3)
    n_hat = np.asarray(quad_normal, dtype=float)

    # Total quad area via diagonal cross-product (exact for planar quads)
    d1 = P[2] - P[0]
    d2 = P[3] - P[1]
    A_total = float(np.linalg.norm(np.cross(d1, d2))) * 0.5
    if A_total < 1e-18:
        return 0.0

    n = n_steel_sub
    sub_area_i = A_total / (n * n)

    # Pre-extract fire-patch centroids for shadow testing (avoids repeated unpacking)
    fire_centroids: list[np.ndarray] = [jp[0] for jp in zone_patches]

    # Pre-parse shadow-beam tuples once outside the hot inner loops
    use_shadow = shadow_beams is not None and len(shadow_beams) > 0
    sb_starts:  list[np.ndarray] = []
    sb_ends:    list[np.ndarray] = []
    sb_radii:   list[float]      = []
    if use_shadow:
        for sb_start, sb_end, sb_radius in shadow_beams:  # type: ignore[misc]
            sb_starts.append(np.asarray(sb_start, dtype=float))
            sb_ends.append(np.asarray(sb_end, dtype=float))
            sb_radii.append(float(sb_radius))

    F_num = 0.0  # accumulates (1/A1) · ΣΣ ... · Ai · Aj  (before /A1 divide)
    for i in range(n):
        s = (i + 0.5) / n   # parametric coordinate along first edge (0..1)
        for j in range(n):
            t = (j + 0.5) / n   # parametric coordinate along second edge (0..1)
            # Bilinear centroid of this steel sub-patch
            c_i = (
                (1 - s) * (1 - t) * P[0]
                + s * (1 - t) * P[1]
                + s * t * P[2]
                + (1 - s) * t * P[3]
            )
            for patch_idx, (j_centroid, j_normal, A_j) in enumerate(zone_patches):
                # Shadow check: skip this pair if any beam blocks the ray
                if use_shadow:
                    blocked = False
                    for k in range(len(sb_starts)):
                        if ray_segment_intersects_beam(
                            c_i, fire_centroids[patch_idx],
                            sb_starts[k], sb_ends[k], sb_radii[k],
                        ):
                            blocked = True
                            break
                    if blocked:
                        continue

                r_vec = j_centroid - c_i
                r = float(np.linalg.norm(r_vec))
                if r < 1e-9:
                    continue
                r_hat = r_vec / r
                cos_i = float(np.dot(n_hat, r_hat))
                cos_j = float(np.dot(j_normal, -r_hat))
                if cos_i <= 0.0 or cos_j <= 0.0:
                    continue
                F_num += cos_i * cos_j / (math.pi * r * r) * sub_area_i * A_j

    return min(F_num / A_total, 1.0)
