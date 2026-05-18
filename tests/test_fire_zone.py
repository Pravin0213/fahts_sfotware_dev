"""Tests for Phase 2.1–2.2: FireCurve and FireZone."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone


# ── FireCurve ─────────────────────────────────────────────────────────────────

class TestFireCurveISO834:
    def test_ambient_at_t0(self):
        fc = FireCurve(FireCurveType.ISO_834)
        assert fc.temperature(0.0) == pytest.approx(20.0)

    def test_known_value_at_1min(self):
        fc = FireCurve(FireCurveType.ISO_834)
        # T = 20 + 345 * log10(8*1+1) = 20 + 345*log10(9)
        expected = 20.0 + 345.0 * math.log10(9.0)
        assert fc.temperature(60.0) == pytest.approx(expected, rel=1e-6)

    def test_known_value_at_30min(self):
        fc = FireCurve(FireCurveType.ISO_834)
        expected = 20.0 + 345.0 * math.log10(8.0 * 30.0 + 1.0)
        assert fc.temperature(30 * 60) == pytest.approx(expected, rel=1e-6)

    def test_monotonically_increasing(self):
        fc = FireCurve(FireCurveType.ISO_834)
        times = [60 * t for t in range(1, 241)]
        temps = [fc.temperature(t) for t in times]
        assert all(t2 > t1 for t1, t2 in zip(temps, temps[1:]))

    def test_custom_ambient(self):
        fc = FireCurve(FireCurveType.ISO_834, T_ambient=0.0)
        assert fc.temperature(0.0) == pytest.approx(0.0)


class TestFireCurveHydrocarbon:
    def test_ambient_at_t0(self):
        fc = FireCurve(FireCurveType.HYDROCARBON)
        assert fc.temperature(0.0) == pytest.approx(20.0, abs=1.0)

    def test_approaches_1100_at_long_time(self):
        fc = FireCurve(FireCurveType.HYDROCARBON)
        T_long = fc.temperature(4 * 3600)
        assert T_long == pytest.approx(20.0 + 1080.0, abs=1.0)

    def test_fast_rise_first_minute(self):
        fc = FireCurve(FireCurveType.HYDROCARBON)
        T1min = fc.temperature(60.0)
        T_iso = FireCurve(FireCurveType.ISO_834).temperature(60.0)
        assert T1min > T_iso, "Hydrocarbon curve should be hotter than ISO 834 at 1 min"

    def test_monotonically_increasing(self):
        fc = FireCurve(FireCurveType.HYDROCARBON)
        times = [60 * t for t in range(1, 241)]
        temps = [fc.temperature(t) for t in times]
        assert all(t2 >= t1 for t1, t2 in zip(temps, temps[1:]))


class TestFireCurveUserDefined:
    def test_exact_points(self):
        pts = [(0.0, 20.0), (60.0, 500.0), (3600.0, 1000.0)]
        fc = FireCurve(FireCurveType.USER_DEFINED, user_points=pts)
        assert fc.temperature(0.0) == pytest.approx(20.0)
        assert fc.temperature(60.0) == pytest.approx(500.0)
        assert fc.temperature(3600.0) == pytest.approx(1000.0)

    def test_linear_interpolation(self):
        pts = [(0.0, 0.0), (100.0, 100.0)]
        fc = FireCurve(FireCurveType.USER_DEFINED, user_points=pts)
        assert fc.temperature(50.0) == pytest.approx(50.0)

    def test_extrapolation_clamps_to_endpoints(self):
        pts = [(100.0, 500.0), (200.0, 600.0)]
        fc = FireCurve(FireCurveType.USER_DEFINED, user_points=pts)
        assert fc.temperature(0.0) == pytest.approx(500.0)    # clamps to start
        assert fc.temperature(300.0) == pytest.approx(600.0)  # clamps to end

    def test_empty_returns_ambient(self):
        fc = FireCurve(FireCurveType.USER_DEFINED, user_points=[])
        assert fc.temperature(60.0) == pytest.approx(20.0)

    def test_unsorted_points_sorted_correctly(self):
        pts = [(100.0, 600.0), (0.0, 20.0), (50.0, 300.0)]
        fc = FireCurve(FireCurveType.USER_DEFINED, user_points=pts)
        assert fc.temperature(25.0) == pytest.approx(160.0)  # midpoint 20-300


# ── FireZone ──────────────────────────────────────────────────────────────────

def _make_zone(cx=0.0, cy=0.0, cz=0.0, dx=2.0, dy=2.0, dz=2.0) -> FireZone:
    return FireZone(
        name="Test Zone",
        center=np.array([cx, cy, cz]),
        dims=np.array([dx, dy, dz]),
    )


class TestFireZoneBounds:
    def test_bounds_symmetric(self):
        zone = _make_zone(cx=10.0, cy=20.0, cz=5.0, dx=4.0, dy=6.0, dz=2.0)
        lo, hi = zone.bounds()
        assert lo == pytest.approx([8.0, 17.0, 4.0])
        assert hi == pytest.approx([12.0, 23.0, 6.0])


class TestFireZoneContains:
    def test_centre_inside(self):
        zone = _make_zone()
        assert zone.contains_point(np.array([0.0, 0.0, 0.0]))

    def test_corner_inside(self):
        zone = _make_zone(dx=2.0, dy=2.0, dz=2.0)
        assert zone.contains_point(np.array([1.0, 1.0, 1.0]))

    def test_just_outside(self):
        zone = _make_zone(dx=2.0, dy=2.0, dz=2.0)
        assert not zone.contains_point(np.array([1.01, 0.0, 0.0]))

    def test_tolerance_extends_boundary(self):
        zone = _make_zone(dx=2.0, dy=2.0, dz=2.0)
        # Without tolerance: 1.05 is outside
        assert not zone.contains_point(np.array([1.05, 0.0, 0.0]))
        # With tolerance of 0.1: now inside
        assert zone.contains_point(np.array([1.05, 0.0, 0.0]), tolerance=0.1)

    def test_inactive_zone_irrelevant_to_contains(self):
        # contains_point is a geometric check — active flag is checked higher up
        zone = _make_zone()
        zone.active = False
        assert zone.contains_point(np.array([0.0, 0.0, 0.0]))


class TestFireZoneTemperature:
    def test_delegates_to_curve(self):
        curve = FireCurve(FireCurveType.ISO_834)
        zone = FireZone(
            name="Z",
            center=np.zeros(3),
            dims=np.ones(3),
            curve=curve,
        )
        expected = curve.temperature(3600.0)
        assert zone.temperature(3600.0) == pytest.approx(expected)
