"""Tests for RadiationBall (USERFLUX type 0) — two-zone spherical heat source."""
import numpy as np
import pytest

from fahts.core.heat.sources.rad_ball import RadiationBall


def _ball(
    center=(0.0, 0.0, 0.0),
    r1=5.0,
    flux1=350_000.0,
    r2=100.0,
    flux2=1_500.0,
) -> RadiationBall:
    return RadiationBall(
        name="test_ball",
        center=np.array(center, dtype=float),
        r1=r1,
        flux1=flux1,
        r2=r2,
        flux2=flux2,
    )


# ── Construction & validation ─────────────────────────────────────────────────

class TestConstruction:
    def test_active_by_default(self):
        assert _ball().active is True

    def test_temperature_returns_ambient(self):
        b = _ball()
        assert b.temperature(0.0) == pytest.approx(20.0)
        assert b.temperature(9999.0) == pytest.approx(20.0)

    def test_bounds_uses_r2(self):
        b = _ball(center=(10.0, 20.0, 30.0), r1=5.0, r2=100.0)
        lo, hi = b.bounds()
        np.testing.assert_allclose(lo, [-90.0, -80.0, -70.0])
        np.testing.assert_allclose(hi, [110.0, 120.0, 130.0])

    def test_r1_must_be_positive(self):
        with pytest.raises(ValueError, match="r1"):
            _ball(r1=0.0)

    def test_r2_must_exceed_r1(self):
        with pytest.raises(ValueError, match="r2"):
            _ball(r1=10.0, r2=5.0)

    def test_r2_equal_r1_raises(self):
        with pytest.raises(ValueError, match="r2"):
            _ball(r1=5.0, r2=5.0)


# ── flux_at ───────────────────────────────────────────────────────────────────

class TestFluxAt:
    def test_at_centre_returns_flux1(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(0.0) == pytest.approx(350_000.0)

    def test_at_r1_boundary_returns_flux1(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(5.0) == pytest.approx(350_000.0)

    def test_just_outside_r1_returns_flux2(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(5.001) == pytest.approx(1_500.0)

    def test_at_r2_boundary_returns_flux2(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(100.0) == pytest.approx(1_500.0)

    def test_beyond_r2_returns_zero(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(100.001) == pytest.approx(0.0)
        assert b.flux_at(9999.0) == pytest.approx(0.0)

    def test_midpoint_in_outer_zone(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        assert b.flux_at(50.0) == pytest.approx(1_500.0)


# ── exposed_element_ids ───────────────────────────────────────────────────────

class TestExposedElementIds:
    def _make_beams(self, midpoints):
        class FakeBeam:
            def __init__(self, eid, mid):
                self.eid = eid
                self._mid = np.array(mid, dtype=float)
                self.n1, self.n2 = eid * 2, eid * 2 + 1

            def midpoint(self, nodes):
                return self._mid

        return {i: FakeBeam(i, mid) for i, mid in enumerate(midpoints)}, {}

    def test_returns_dict(self):
        b = _ball()
        beams, nodes = self._make_beams([[0.0, 0.0, 0.0]])
        assert isinstance(b.exposed_element_ids(beams, nodes), dict)

    def test_centre_element_gets_flux1(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([[0.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result
        assert result[0] == pytest.approx(350_000.0)

    def test_outer_zone_element_gets_flux2(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([[50.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result
        assert result[0] == pytest.approx(1_500.0)

    def test_element_beyond_r2_not_exposed(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([[200.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 not in result

    def test_element_on_r1_boundary_gets_flux1(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([[5.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert result[0] == pytest.approx(350_000.0)

    def test_element_on_r2_boundary_gets_flux2(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([[100.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert result[0] == pytest.approx(1_500.0)

    def test_mixed_zones(self):
        b = _ball(r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0)
        beams, nodes = self._make_beams([
            [0.0, 0.0, 0.0],    # inner zone → eid 0
            [50.0, 0.0, 0.0],   # outer zone → eid 1
            [200.0, 0.0, 0.0],  # beyond r2  → eid 2 (not exposed)
        ])
        result = b.exposed_element_ids(beams, nodes)
        assert result[0] == pytest.approx(350_000.0)
        assert result[1] == pytest.approx(1_500.0)
        assert 2 not in result

    def test_inactive_ball_still_exposes(self):
        """exposed_element_ids is geometry-only; caller filters active status."""
        b = RadiationBall(
            name="off", center=np.zeros(3),
            r1=5.0, flux1=350_000.0, r2=100.0, flux2=1_500.0,
            active=False,
        )
        beams, nodes = self._make_beams([[0.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result

    def test_benchmark_params(self):
        """Replicate USFOS benchmark: center=(343,484,64), r1=5, r2=100."""
        b = RadiationBall(
            name="benchmark",
            center=np.array([343.0, 484.0, 64.0]),
            r1=5.0,   flux1=350_000.0,
            r2=100.0, flux2=1_500.0,
        )
        assert b.flux_at(0.0)   == pytest.approx(350_000.0)
        assert b.flux_at(5.0)   == pytest.approx(350_000.0)
        assert b.flux_at(5.01)  == pytest.approx(1_500.0)
        assert b.flux_at(100.0) == pytest.approx(1_500.0)
        assert b.flux_at(100.1) == pytest.approx(0.0)
