"""Tests for RadiationBall — exact point-to-sphere flux (Lambertian sphere source)."""
import numpy as np
import pytest

from fahts.core.heat.sources.rad_ball import RadiationBall


def _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0) -> RadiationBall:
    return RadiationBall(
        name="test_ball",
        center=np.array(center, dtype=float),
        radius=radius,
        flux=flux,
    )


# ── Construction & validation ─────────────────────────────────────────────────

class TestConstruction:
    def test_active_by_default(self):
        assert _ball().active is True

    def test_temperature_returns_ambient(self):
        b = _ball()
        assert b.temperature(0.0) == pytest.approx(20.0)
        assert b.temperature(9999.0) == pytest.approx(20.0)

    def test_bounds_uses_radius(self):
        b = _ball(center=(10.0, 20.0, 30.0), radius=5.0)
        lo, hi = b.bounds()
        np.testing.assert_allclose(lo, [5.0, 15.0, 25.0])
        np.testing.assert_allclose(hi, [15.0, 25.0, 35.0])

    def test_radius_must_be_positive(self):
        with pytest.raises(ValueError, match="radius"):
            _ball(radius=0.0)

    def test_flux_must_be_positive(self):
        with pytest.raises(ValueError, match="flux"):
            _ball(flux=0.0)


# ── max_flux_at_distance ────────────────────────────────────────────────────

class TestMaxFluxAtDistance:
    def test_at_surface_returns_flux(self):
        b = _ball(radius=5.0, flux=350_000.0)
        assert b.max_flux_at_distance(5.0) == pytest.approx(350_000.0)

    def test_inside_radius_clamped_at_flux(self):
        b = _ball(radius=5.0, flux=350_000.0)
        assert b.max_flux_at_distance(2.5) == pytest.approx(350_000.0)
        assert b.max_flux_at_distance(0.0) == pytest.approx(350_000.0)

    def test_exterior_follows_inverse_square(self):
        b = _ball(radius=5.0, flux=350_000.0)
        # F = (R/d)^2 exactly (sphere source theorem)
        assert b.max_flux_at_distance(10.0) == pytest.approx(350_000.0 * (5.0 / 10.0) ** 2)
        assert b.max_flux_at_distance(50.0) == pytest.approx(350_000.0 * (5.0 / 50.0) ** 2)

    def test_decays_toward_zero_but_never_hits_it(self):
        b = _ball(radius=5.0, flux=350_000.0)
        assert b.max_flux_at_distance(1.0e6) > 0.0
        assert b.max_flux_at_distance(1.0e6) < 1.0e-3


# ── incident_flux (directional) ─────────────────────────────────────────────

class TestIncidentFlux:
    def test_engulfed_ignores_normal(self):
        """d <= radius: flux applies uniformly regardless of face orientation."""
        b = _ball(radius=5.0, flux=350_000.0)
        point = np.array([2.5, 0.0, 0.0])
        for normal in (
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ):
            assert b.incident_flux(point, normal) == pytest.approx(350_000.0)

    def test_directly_facing_receiver_gets_full_inverse_square_value(self):
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([10.0, 0.0, 0.0])
        normal = np.array([-1.0, 0.0, 0.0])   # points back toward ball centre
        expected = 350_000.0 * (5.0 / 10.0) ** 2
        assert b.incident_flux(point, normal) == pytest.approx(expected)

    def test_face_pointing_away_gets_zero(self):
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([10.0, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])    # points away from ball centre
        assert b.incident_flux(point, normal) == pytest.approx(0.0)

    def test_tangential_face_gets_zero(self):
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([10.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])    # perpendicular to line-of-sight
        assert b.incident_flux(point, normal) == pytest.approx(0.0)

    def test_oblique_face_scales_by_cosine(self):
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([10.0, 0.0, 0.0])
        normal = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)  # 45 deg off
        expected = 350_000.0 * (5.0 / 10.0) ** 2 * np.cos(np.pi / 4)
        assert b.incident_flux(point, normal) == pytest.approx(expected)

    def test_no_normal_returns_direction_agnostic_bound(self):
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([10.0, 0.0, 0.0])
        assert b.incident_flux(point, normal=None) == pytest.approx(
            b.max_flux_at_distance(10.0)
        )

    def test_exact_at_boundary_facing_center_gives_full_flux(self):
        """Continuity check: at d==radius facing the centre, exterior formula == flux."""
        b = _ball(center=(0.0, 0.0, 0.0), radius=5.0, flux=350_000.0)
        point = np.array([5.0, 0.0, 0.0])
        normal = np.array([-1.0, 0.0, 0.0])
        assert b.incident_flux(point, normal) == pytest.approx(350_000.0)


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

    def test_engulfed_element_gets_full_flux(self):
        b = _ball(radius=5.0, flux=350_000.0)
        beams, nodes = self._make_beams([[2.5, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result
        assert result[0] == pytest.approx(350_000.0)

    def test_exterior_element_gets_inverse_square_flux(self):
        b = _ball(radius=5.0, flux=350_000.0)
        beams, nodes = self._make_beams([[50.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result
        assert result[0] == pytest.approx(350_000.0 * (5.0 / 50.0) ** 2)

    def test_far_element_excluded_by_min_flux(self):
        b = _ball(radius=5.0, flux=350_000.0)
        beams, nodes = self._make_beams([[100_000.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes, min_flux=1.0)
        assert 0 not in result

    def test_mixed_distances(self):
        b = _ball(radius=5.0, flux=350_000.0)
        beams, nodes = self._make_beams([
            [2.5, 0.0, 0.0],       # engulfed → eid 0
            [50.0, 0.0, 0.0],      # exterior → eid 1
            [1_000_000.0, 0.0, 0.0],  # negligible → eid 2 (not exposed)
        ])
        result = b.exposed_element_ids(beams, nodes)
        assert result[0] == pytest.approx(350_000.0)
        assert result[1] == pytest.approx(350_000.0 * (5.0 / 50.0) ** 2)
        assert 2 not in result

    def test_inactive_ball_still_exposes(self):
        """exposed_element_ids is geometry-only; caller filters active status."""
        b = RadiationBall(
            name="off", center=np.zeros(3), radius=5.0, flux=350_000.0, active=False,
        )
        beams, nodes = self._make_beams([[0.0, 0.0, 0.0]])
        result = b.exposed_element_ids(beams, nodes)
        assert 0 in result
