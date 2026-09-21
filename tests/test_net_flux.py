"""Tests for Phase 2.5–2.6: net_flux and view_factor modules."""
from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.bc.net_flux import (
    _SIGMA,
    convective_flux,
    net_heat_flux,
    radiative_flux,
)
from fahts.core.heat.bc.view_factor import (
    ElementHeatBC,
    compute_all_bcs,
    compute_element_bc,
    exposure_flags,
    exposed_element_ids,
)
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement
from fahts.core.model.node import Node


# ── net_heat_flux ─────────────────────────────────────────────────────────────

class TestNetHeatFlux:
    def test_zero_when_equal_temperatures(self):
        q = net_heat_flux(T_fire=500.0, T_steel=500.0, epsilon_m=0.7, h_conv=25.0)
        assert q == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_fire_hotter(self):
        q = net_heat_flux(T_fire=800.0, T_steel=20.0, epsilon_m=0.7, h_conv=25.0)
        assert q > 0.0

    def test_negative_when_steel_hotter(self):
        q = net_heat_flux(T_fire=20.0, T_steel=800.0, epsilon_m=0.7, h_conv=25.0)
        assert q < 0.0

    def test_radiative_component_matches_formula(self):
        Tf, Ts = 800.0, 20.0
        eps = 0.7
        q_rad_expected = eps * _SIGMA * ((Tf + 273.15)**4 - (Ts + 273.15)**4)
        assert radiative_flux(Tf, Ts, eps) == pytest.approx(q_rad_expected, rel=1e-10)

    def test_convective_component_matches_formula(self):
        Tf, Ts, h = 800.0, 20.0, 25.0
        assert convective_flux(Tf, Ts, h) == pytest.approx(h * (Tf - Ts), rel=1e-10)

    def test_net_equals_rad_plus_conv(self):
        Tf, Ts, eps, h = 600.0, 100.0, 0.5, 25.0
        q_net = net_heat_flux(Tf, Ts, eps, h)
        q_rad = radiative_flux(Tf, Ts, eps)
        q_conv = convective_flux(Tf, Ts, h)
        assert q_net == pytest.approx(q_rad + q_conv, rel=1e-10)

    def test_en1993_reference_value(self):
        """Spot-check: EN 1993-1-2 §A.1 example (approximate)."""
        # T_fire = 945°C (ISO 834 at 60 min), T_steel = 20°C, eps=0.7, h=25
        Tf = 945.0
        Ts = 20.0
        eps = 0.7
        h = 25.0
        q = net_heat_flux(Tf, Ts, eps, h)
        # Radiative dominates at high T; expect order of ~100 kW/m²
        assert 50_000 < q < 300_000, f"Unexpected q_net={q:.0f} W/m²"

    def test_epsilon_zero_gives_only_convection(self):
        Tf, Ts, h = 800.0, 20.0, 25.0
        q = net_heat_flux(Tf, Ts, 0.0, h)
        assert q == pytest.approx(h * (Tf - Ts), rel=1e-10)

    def test_h_conv_zero_gives_only_radiation(self):
        Tf, Ts, eps = 800.0, 20.0, 0.7
        q = net_heat_flux(Tf, Ts, eps, 0.0)
        assert q == pytest.approx(radiative_flux(Tf, Ts, eps), rel=1e-10)


# ── view_factor / exposure_flags ──────────────────────────────────────────────

def _make_beam(mid: tuple[float, float, float]) -> tuple[BeamElement, dict]:
    """Return a beam whose midpoint is at *mid* and a nodes dict."""
    x, y, z = mid
    n1 = Node(nid=1, x=x - 0.5, y=y, z=z)
    n2 = Node(nid=2, x=x + 0.5, y=y, z=z)
    direction = np.array([1.0, 0.0, 0.0])
    beam = BeamElement(
        eid=1, n1=1, n2=2,
        mat_id=1, geom_id=1, lcoor_id=1,
        length=1.0,
        direction=direction,
        local_z=np.array([0.0, 0.0, 1.0]),
    )
    return beam, {1: n1, 2: n2}


def _zone_at(cx=0.0, cy=0.0, cz=0.0, d=4.0) -> FireZone:
    return FireZone(
        name="Z",
        center=np.array([cx, cy, cz]),
        dims=np.array([d, d, d]),
    )


class TestExposureFlags:
    def test_inside_zone_all_faces_exposed(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone = _zone_at()
        flags = exposure_flags(beam, [zone], nodes)
        assert all(v == 1.0 for v in flags.values())

    def test_outside_zone_no_faces_exposed(self):
        beam, nodes = _make_beam((10.0, 10.0, 10.0))
        zone = _zone_at()
        flags = exposure_flags(beam, [zone], nodes)
        assert all(v == 0.0 for v in flags.values())

    def test_inactive_zone_not_counted(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone = _zone_at()
        zone.active = False
        flags = exposure_flags(beam, [zone], nodes)
        assert all(v == 0.0 for v in flags.values())

    def test_multiple_zones_one_matches(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone_far = _zone_at(cx=100.0)
        zone_near = _zone_at(cx=0.0)
        flags = exposure_flags(beam, [zone_far, zone_near], nodes)
        assert all(v == 1.0 for v in flags.values())

    def test_tolerance_extends_zone(self):
        # midpoint (2.6) and n1 (2.1) are both strictly outside ±2.0 half-extent
        beam, nodes = _make_beam((2.6, 0.0, 0.0))
        zone = _zone_at(d=4.0)  # extends to ±2.0 in each direction
        assert all(v == 0.0 for v in exposure_flags(beam, [zone], nodes).values())
        assert all(v == 1.0 for v in exposure_flags(beam, [zone], nodes, tolerance=0.6).values())


class TestExposedElementIds:
    def _make_model_elements(self) -> tuple[dict, dict]:
        """Two elements: one inside zone at origin, one far away."""
        n_in = {
            1: Node(1, -0.5, 0.0, 0.0),
            2: Node(2,  0.5, 0.0, 0.0),
        }
        n_out = {
            10: Node(10, 99.5, 0.0, 0.0),
            11: Node(11, 100.5, 0.0, 0.0),
        }
        nodes = {**n_in, **n_out}
        e_in = BeamElement(
            eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
            length=1.0, direction=np.array([1.0, 0.0, 0.0]),
            local_z=np.array([0.0, 0.0, 1.0]),
        )
        e_out = BeamElement(
            eid=2, n1=10, n2=11, mat_id=1, geom_id=1, lcoor_id=1,
            length=1.0, direction=np.array([1.0, 0.0, 0.0]),
            local_z=np.array([0.0, 0.0, 1.0]),
        )
        elements = {1: e_in, 2: e_out}
        return elements, nodes

    def test_correct_elements_exposed(self):
        elements, nodes = self._make_model_elements()
        zone = _zone_at(cx=0.0, d=4.0)
        eids = exposed_element_ids(elements, [zone], nodes)
        assert eids == {1}

    def test_no_zones_returns_empty(self):
        elements, nodes = self._make_model_elements()
        assert exposed_element_ids(elements, [], nodes) == set()

    def test_all_inactive_returns_empty(self):
        elements, nodes = self._make_model_elements()
        zone = _zone_at()
        zone.active = False
        assert exposed_element_ids(elements, [zone], nodes) == set()


class TestComputeElementBC:
    def test_outside_zone_returns_none(self):
        beam, nodes = _make_beam((100.0, 0.0, 0.0))
        zone = _zone_at()
        bc = compute_element_bc(beam, [zone], nodes, t=0.0)
        assert bc is None

    def test_inside_zone_returns_bc(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone = _zone_at()
        bc = compute_element_bc(beam, [zone], nodes, t=60.0)
        assert bc is not None
        assert bc.eid == 1
        assert bc.t == 60.0
        assert bc.is_exposed

    def test_all_faces_have_equal_flux_uniform_exposure(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone = _zone_at()
        bc = compute_element_bc(beam, [zone], nodes, t=60.0)
        assert bc is not None
        fluxes = list(bc.face_fluxes.values())
        assert all(f == pytest.approx(fluxes[0]) for f in fluxes)

    def test_total_flux_positive(self):
        beam, nodes = _make_beam((0.0, 0.0, 0.0))
        zone = _zone_at()
        bc = compute_element_bc(beam, [zone], nodes, t=3600.0)
        assert bc is not None
        assert bc.total_flux > 0.0


class TestComputeAllBCs:
    def test_returns_only_exposed(self):
        n = {
            1: Node(1, -0.5, 0.0, 0.0),
            2: Node(2,  0.5, 0.0, 0.0),
            3: Node(3, 99.5, 0.0, 0.0),
            4: Node(4, 100.5, 0.0, 0.0),
        }
        e_in = BeamElement(eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
                           length=1.0, direction=np.array([1.0, 0.0, 0.0]),
                           local_z=np.array([0.0, 0.0, 1.0]))
        e_out = BeamElement(eid=2, n1=3, n2=4, mat_id=1, geom_id=1, lcoor_id=1,
                            length=1.0, direction=np.array([1.0, 0.0, 0.0]),
                            local_z=np.array([0.0, 0.0, 1.0]))
        elements = {1: e_in, 2: e_out}
        zone = _zone_at(cx=0.0, d=4.0)
        bcs = compute_all_bcs(elements, [zone], n, t=60.0)
        assert len(bcs) == 1
        assert bcs[0].eid == 1

    def test_no_zones_returns_empty(self):
        n = {1: Node(1, 0.0, 0.0, 0.0), 2: Node(2, 1.0, 0.0, 0.0)}
        e = BeamElement(eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
                        length=1.0, direction=np.array([1.0, 0.0, 0.0]),
                        local_z=np.array([0.0, 0.0, 1.0]))
        bcs = compute_all_bcs({1: e}, [], n, t=60.0)
        assert bcs == []
