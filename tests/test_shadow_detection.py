"""
Tests for ray-based obstruction/shadow detection in radiative exchange.

Covers:
1. No beams → no shadowing (all pairs visible).
2. A beam directly between a quad centroid and a fire patch → fully shadowed.
3. A beam beside the ray but not intersecting → not shadowed.
4. Self-element exclusion: the element whose surface produced the quad centroid
   does not block the ray to the fire patch when its ends are at the same
   position as the ray origin / ray target.
5. Multi-beam scene: only the beam in the direct line-of-sight causes shadowing;
   others beside it do not.
6. geometric_view_factor_double_area with shadow_beams=None is unchanged
   (backward compatibility).
7. geometric_view_factor_double_area with a blocking beam returns zero (or
   near-zero when using sub-patches).
8. compute_shadow_mask shape and dtype are correct.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fahts.core.heat.bc.view_factor import (
    compute_shadow_mask,
    geometric_view_factor_double_area,
    ray_segment_intersects_beam,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quad_at_origin() -> np.ndarray:
    """A small 0.1×0.1 m quad centred at the origin in the z=0 plane."""
    return np.array([
        [-0.05, -0.05, 0.0],
        [ 0.05, -0.05, 0.0],
        [ 0.05,  0.05, 0.0],
        [-0.05,  0.05, 0.0],
    ], dtype=float)


def _fire_patch_above(z: float = 5.0) -> tuple[np.ndarray, np.ndarray, float]:
    """A small fire patch centred directly above the quad."""
    return (np.array([0.0, 0.0, z]), np.array([0.0, 0.0, -1.0]), 0.25)


# ── Test 1: No beams → no shadowing ──────────────────────────────────────────

def test_no_beams_no_shadowing():
    """With an empty beam list every quad–patch pair is visible."""
    quad_centroids = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=float)
    fire_centroids = np.array([
        [0.5, 0.0, 5.0],
        [0.5, 1.0, 5.0],
    ], dtype=float)
    beam_starts = np.zeros((0, 3), dtype=float)
    beam_ends   = np.zeros((0, 3), dtype=float)
    beam_radii  = np.zeros(0, dtype=float)

    mask = compute_shadow_mask(
        quad_centroids, fire_centroids, beam_starts, beam_ends, beam_radii
    )

    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask.all(), "No beams → all pairs must be visible"


# ── Test 2: Beam directly between quad and fire patch → fully shadowed ────────

def test_beam_directly_between_quad_and_patch_is_shadowed():
    """A beam placed exactly on the line of sight blocks the ray."""
    # Quad at origin, fire patch 6 m above.  Beam centred at z=3, large radius.
    quad_centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)
    fire_centroids = np.array([[0.0, 0.0, 6.0]], dtype=float)

    # Beam runs parallel to X, centred at (0, 0, 3) — directly in the ray path
    beam_starts = np.array([[-1.0, 0.0, 3.0]], dtype=float)
    beam_ends   = np.array([[ 1.0, 0.0, 3.0]], dtype=float)
    beam_radii  = np.array([0.5], dtype=float)   # large radius → definitely blocks

    mask = compute_shadow_mask(
        quad_centroids, fire_centroids, beam_starts, beam_ends, beam_radii
    )

    assert mask.shape == (1, 1)
    assert not mask[0, 0], "Beam in direct line-of-sight must shadow the pair"


# ── Test 3: Beam beside the ray → not shadowed ────────────────────────────────

def test_beam_beside_ray_not_shadowed():
    """A beam displaced laterally from the ray does not block it."""
    quad_centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)
    fire_centroids = np.array([[0.0, 0.0, 6.0]], dtype=float)

    # Beam offset 2 m in Y, radius only 0.3 m — ray travels along X=0, Y=0
    beam_starts = np.array([[-1.0, 2.0, 3.0]], dtype=float)
    beam_ends   = np.array([[ 1.0, 2.0, 3.0]], dtype=float)
    beam_radii  = np.array([0.3], dtype=float)

    mask = compute_shadow_mask(
        quad_centroids, fire_centroids, beam_starts, beam_ends, beam_radii
    )

    assert mask[0, 0], "Beam beside the ray must NOT shadow the pair"


# ── Test 4: Self-element exclusion ────────────────────────────────────────────

def test_self_element_does_not_shadow():
    """
    The beam that owns the steel quad is listed as a blocker but its ends
    are at z=0 — the same plane as the quad centroid.  The capsule intersection
    test uses an epsilon guard so a 'beam' at the ray origin does not falsely
    block its own radiation.

    Specifically, the beam runs horizontally (parallel to the XY plane) and the
    ray goes vertically upward.  The closest approach on the ray occurs at s=0
    (the origin itself), which is outside the strict (EPS, 1-EPS) window.
    """
    quad_centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)
    fire_centroids = np.array([[0.0, 0.0, 5.0]], dtype=float)

    # "Self" beam: same span as the steel element, z=0 plane, large radius
    beam_starts = np.array([[-1.0, 0.0, 0.0]], dtype=float)
    beam_ends   = np.array([[ 1.0, 0.0, 0.0]], dtype=float)
    beam_radii  = np.array([0.15], dtype=float)

    result = ray_segment_intersects_beam(
        quad_centroids[0], fire_centroids[0],
        beam_starts[0], beam_ends[0], beam_radii[0],
    )
    # The ray goes straight up; the closest point on the ray to the beam is at
    # the very origin (s≈0), which is excluded by the epsilon guard.
    assert not result, "Self-element at ray origin must not shadow the ray"


# ── Test 5: Multi-beam scene ──────────────────────────────────────────────────

def test_multi_beam_only_blocking_beam_causes_shadow():
    """
    Three beams: one blocks the ray, two are beside it.
    Only the pair involving the fire patch behind the blocking beam is shadowed.
    """
    quad_centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)
    # Two fire patches: one directly above (blocked), one offset to the side (clear)
    fire_centroids = np.array([
        [0.0, 0.0, 6.0],   # directly above → blocked by beam 0
        [4.0, 0.0, 6.0],   # far to the side → not blocked
    ], dtype=float)

    # Beam 0: in the direct line of sight (z=3, centred on X=0)
    # Beam 1: beside both rays (Y=3)
    # Beam 2: on the far side (X=4, z=3) — in the path to fire_centroids[1]
    #          but with a small radius that misses the actual ray
    beam_starts = np.array([
        [-1.0, 0.0, 3.0],   # beam 0 — blocking
        [-1.0, 3.0, 3.0],   # beam 1 — beside both rays
        [ 3.0, 0.0, 3.0],   # beam 2 — near patch 1 ray but small radius
    ], dtype=float)
    beam_ends = np.array([
        [ 1.0, 0.0, 3.0],
        [ 1.0, 3.0, 3.0],
        [ 5.0, 0.0, 3.0],
    ], dtype=float)
    beam_radii = np.array([0.5, 0.3, 0.05], dtype=float)

    mask = compute_shadow_mask(
        quad_centroids, fire_centroids, beam_starts, beam_ends, beam_radii
    )

    assert not mask[0, 0], "Pair (quad0, patch0) must be shadowed by beam 0"
    assert mask[0, 1], "Pair (quad0, patch1) must NOT be shadowed"


# ── Test 6: geometric_view_factor_double_area backward compatibility ──────────

def test_gvf_double_area_shadow_none_unchanged():
    """
    Passing shadow_beams=None (default) gives the same result as before.
    Regression: the shadow parameter must not alter the base view-factor value.
    """
    quad    = _quad_at_origin()
    normal  = np.array([0.0, 0.0, 1.0])
    patches = [_fire_patch_above(5.0)]

    f_no_shadow  = geometric_view_factor_double_area(quad, normal, patches, n_steel_sub=2)
    f_none_param = geometric_view_factor_double_area(
        quad, normal, patches, n_steel_sub=2, shadow_beams=None
    )
    f_empty_list = geometric_view_factor_double_area(
        quad, normal, patches, n_steel_sub=2, shadow_beams=[]
    )

    assert f_no_shadow  > 0.0
    assert f_none_param == pytest.approx(f_no_shadow, rel=1e-12)
    assert f_empty_list == pytest.approx(f_no_shadow, rel=1e-12)


# ── Test 7: geometric_view_factor_double_area with blocking beam → near zero ──

def test_gvf_double_area_blocking_beam_reduces_view_factor():
    """
    Inserting a large-radius beam directly between the quad and the fire patch
    should reduce (ideally zero out) the view factor.
    """
    quad   = _quad_at_origin()
    normal = np.array([0.0, 0.0, 1.0])
    patch  = _fire_patch_above(5.0)

    # No shadow
    f_open = geometric_view_factor_double_area(quad, normal, [patch], n_steel_sub=2)

    # Blocking beam at mid-height with a radius larger than the quad half-width
    shadow_beams = [
        (np.array([-1.0, 0.0, 2.5]), np.array([1.0, 0.0, 2.5]), 1.0)
    ]
    f_blocked = geometric_view_factor_double_area(
        quad, normal, [patch], n_steel_sub=2, shadow_beams=shadow_beams
    )

    assert f_open > 0.0, "Unshadowed view factor must be positive"
    assert f_blocked < f_open, "Blocked view factor must be less than unshadowed"


# ── Test 8: compute_shadow_mask return shape and dtype ────────────────────────

def test_compute_shadow_mask_return_type():
    """compute_shadow_mask always returns a (n_q, n_p) bool array."""
    quad_centroids = np.random.default_rng(0).random((5, 3))
    fire_centroids = np.random.default_rng(1).random((3, 3)) + np.array([0, 0, 10])

    # With one blocking beam far from any ray
    beam_starts = np.array([[100.0, 0.0, 0.0]], dtype=float)
    beam_ends   = np.array([[101.0, 0.0, 0.0]], dtype=float)
    beam_radii  = np.array([0.01], dtype=float)

    mask = compute_shadow_mask(
        quad_centroids, fire_centroids, beam_starts, beam_ends, beam_radii
    )

    assert mask.shape == (5, 3)
    assert mask.dtype == bool
