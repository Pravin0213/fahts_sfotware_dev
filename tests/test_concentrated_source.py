"""
Tests for §3.5.4 concentrated (point) heat source.

Covers:
  - ConcentratedSource dataclass construction
  - flux_at: direct front (theta=0) matches E/(4*pi*r^2)
  - flux_at: theta=90° gives zero (cos=0)
  - flux_at: behind source (theta=180°) gives zero (clamped)
  - Inverse-square falloff: flux ∝ 1/r^2
  - per_quad_flux vectorised form matches scalar flux_at
  - Callable (time-varying) power works correctly
  - AnalysisConfig.concentrated_sources field exists and validates
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fahts.core.heat.sources.concentrated_source import ConcentratedSource
from fahts.core.results.analysis_config import AnalysisConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cs(power: float = 1000.0) -> ConcentratedSource:
    return ConcentratedSource(
        name="test_cs",
        center=np.array([0.0, 0.0, 0.0]),
        power=power,
    )


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:
    def test_constant_power(self):
        cs = _make_cs(500.0)
        assert cs.emitted_power(0.0) == pytest.approx(500.0)
        assert cs.emitted_power(100.0) == pytest.approx(500.0)

    def test_callable_power(self):
        cs = ConcentratedSource(
            name="ramp",
            center=np.zeros(3),
            power=lambda t: 100.0 * t,
        )
        assert cs.emitted_power(0.0) == pytest.approx(0.0)
        assert cs.emitted_power(5.0) == pytest.approx(500.0)

    def test_temperature_returns_ambient(self):
        cs = _make_cs()
        assert cs.temperature(0.0) == pytest.approx(20.0)

    def test_bounds_is_point(self):
        cs = _make_cs()
        lo, hi = cs.bounds()
        np.testing.assert_array_equal(lo, cs.center)
        np.testing.assert_array_equal(hi, cs.center)

    def test_active_default_true(self):
        cs = _make_cs()
        assert cs.active is True


# ── flux_at physics ───────────────────────────────────────────────────────────

class TestFluxAt:
    """flux_at(point, normal, t) tests."""

    def test_direct_front_matches_inverse_square(self):
        """At theta=0, q = E/(4*pi*r^2)."""
        E = 1000.0
        r = 3.0
        cs = ConcentratedSource(name="s", center=np.zeros(3), power=E)
        # Point directly in front (+x), normal pointing back toward source (−x)
        point  = np.array([r, 0.0, 0.0])
        normal = np.array([-1.0, 0.0, 0.0])   # outward normal points toward source
        expected = E / (4.0 * math.pi * r * r)
        assert cs.flux_at(point, normal, t=0.0) == pytest.approx(expected, rel=1e-9)

    def test_orthogonal_gives_zero(self):
        """At theta=90° (normal perpendicular to ray), flux is zero."""
        cs = _make_cs(1000.0)
        point  = np.array([3.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])   # perpendicular to source ray
        assert cs.flux_at(point, normal, t=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_behind_source_gives_zero(self):
        """Back-facing surface (cos<0) receives zero flux."""
        cs = _make_cs(1000.0)
        point  = np.array([3.0, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])   # outward normal points away from source
        assert cs.flux_at(point, normal, t=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_inverse_square_falloff(self):
        """Flux halves when distance doubles (cos fixed)."""
        E = 5000.0
        cs = ConcentratedSource(name="s", center=np.zeros(3), power=E)
        normal = np.array([-1.0, 0.0, 0.0])
        q1 = cs.flux_at(np.array([2.0, 0.0, 0.0]), normal, t=0.0)
        q2 = cs.flux_at(np.array([4.0, 0.0, 0.0]), normal, t=0.0)
        assert q1 / q2 == pytest.approx(4.0, rel=1e-9)

    def test_no_normal_returns_omnidirectional(self):
        """flux_at with normal=None returns E/(4*pi*r^2) with no cosine term."""
        E = 2000.0
        r = 5.0
        cs = ConcentratedSource(name="s", center=np.zeros(3), power=E)
        q = cs.flux_at(np.array([r, 0.0, 0.0]), normal=None, t=0.0)
        expected = E / (4.0 * math.pi * r * r)
        assert q == pytest.approx(expected, rel=1e-9)

    def test_callable_power_at_time(self):
        """Time-varying power is evaluated correctly."""
        cs = ConcentratedSource(
            name="ramp", center=np.zeros(3), power=lambda t: 100.0 * t
        )
        r = 2.0
        normal = np.array([-1.0, 0.0, 0.0])
        t = 10.0
        E = 100.0 * t
        expected = E / (4.0 * math.pi * r * r)
        assert cs.flux_at(np.array([r, 0.0, 0.0]), normal, t=t) == pytest.approx(expected, rel=1e-9)

    def test_source_at_point_returns_zero(self):
        """r=0 does not raise and returns zero."""
        cs = _make_cs(1000.0)
        assert cs.flux_at(np.zeros(3), np.array([1.0, 0.0, 0.0]), t=0.0) == pytest.approx(0.0)


# ── per_quad_flux vectorised ──────────────────────────────────────────────────

class TestPerQuadFlux:
    def test_matches_scalar_flux_at(self):
        """per_quad_flux and flux_at must agree element-by-element."""
        E = 3000.0
        cs = ConcentratedSource(name="s", center=np.zeros(3), power=E)
        points  = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        normals = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        q_vec = cs.per_quad_flux(points, normals, t=0.0)
        for i, (pt, n) in enumerate(zip(points, normals)):
            assert q_vec[i] == pytest.approx(cs.flux_at(pt, n, t=0.0), rel=1e-9)

    def test_back_facing_quads_zero(self):
        """Quads facing away from source receive zero flux."""
        cs = _make_cs(1000.0)
        pts = np.array([[3.0, 0.0, 0.0]])
        nrm = np.array([[1.0, 0.0, 0.0]])   # away from source
        q = cs.per_quad_flux(pts, nrm, t=0.0)
        assert q[0] == pytest.approx(0.0, abs=1e-12)

    def test_shape_correct(self):
        n = 5
        cs = _make_cs(1000.0)
        pts = np.random.randn(n, 3) + np.array([10.0, 0.0, 0.0])
        nrm = np.tile([-1.0, 0.0, 0.0], (n, 1))
        q = cs.per_quad_flux(pts, nrm, t=0.0)
        assert q.shape == (n,)


# ── AnalysisConfig integration ────────────────────────────────────────────────

class TestAnalysisConfig:
    def test_default_empty_list(self):
        cfg = AnalysisConfig(t_end=60.0, dt=10.0, output_dt=10.0)
        assert cfg.concentrated_sources == []

    def test_accepts_concentrated_source(self):
        cs = ConcentratedSource(
            name="cs1", center=np.array([0.0, 0.0, 0.0]), power=500.0
        )
        cfg = AnalysisConfig(
            t_end=60.0, dt=10.0, output_dt=10.0,
            concentrated_sources=[cs],
        )
        assert len(cfg.concentrated_sources) == 1
        assert cfg.concentrated_sources[0] is cs
