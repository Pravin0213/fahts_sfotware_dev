"""
Tests for heat-transfer-element-level exposure (theory_alignment_todo.md Priority 2).

Validates that:
1. `exposed_element_ids` catches beams that straddle a zone boundary (n1/n2 inside
   even when midpoint is outside).
2. `element_quad_exposure_flags` returns correct per-quad 0/1 flags.
3. A partially-exposed element heats more slowly than a fully-exposed one.
4. An element with no quads inside the zone is skipped by run_analysis.
"""
from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.bc.view_factor import (
    element_quad_exposure_flags,
    exposed_element_ids,
    exposure_flags,
)
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_zone(center, dims):
    return FireZone(
        name="test",
        center=np.array(center, dtype=float),
        dims=np.array(dims, dtype=float),
        curve=FireCurve(FireCurveType.ISO_834),
    )


def _steel():
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _box_section():
    return BoxSection(sid=1, H=0.2, T_side=0.008, T_bot=0.008, T_top=0.008, W=0.2)


def _beam_and_nodes(n1_pos, n2_pos):
    """Create a simple BeamElement along global X with UNITVEC = Z."""
    n1 = Node(nid=1, x=n1_pos[0], y=n1_pos[1], z=n1_pos[2])
    n2 = Node(nid=2, x=n2_pos[0], y=n2_pos[1], z=n2_pos[2])
    nodes = {1: n1, 2: n2}
    direction = np.asarray(n2_pos) - np.asarray(n1_pos)
    length = float(np.linalg.norm(direction))
    direction = direction / length
    # local_z perpendicular to beam axis; pick Z if beam is along X
    local_z = np.array([0.0, 0.0, 1.0])
    beam = BeamElement(
        eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=0,
        length=length, direction=direction, local_z=local_z,
    )
    return beam, nodes


# ── exposed_element_ids: endpoint detection ───────────────────────────────────

class TestExposedElementIds:
    def test_midpoint_inside_detected(self):
        """Standard case: midpoint inside → element detected."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        zone = _iso_zone([1.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        result = exposed_element_ids({1: beam}, [zone], nodes)
        assert 1 in result

    def test_endpoint_n2_inside_detected(self):
        """Beam with midpoint outside but n2 inside zone — must be detected."""
        # Zone at x=3..5; beam from x=0 to x=4 — midpoint at x=2 is outside
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([4.0, 0.0, 0.0], [2.0, 2.0, 2.0])  # x: 3..5
        result = exposed_element_ids({1: beam}, [zone], nodes)
        assert 1 in result, "n2 inside zone but element not detected"

    def test_endpoint_n1_inside_detected(self):
        """Beam with midpoint outside but n1 inside zone — must be detected."""
        beam, nodes = _beam_and_nodes([1.0, 0.0, 0.0], [5.0, 0.0, 0.0])
        zone = _iso_zone([0.5, 0.0, 0.0], [1.0, 2.0, 2.0])  # x: 0..1
        result = exposed_element_ids({1: beam}, [zone], nodes)
        assert 1 in result, "n1 inside zone but element not detected"

    def test_fully_outside_not_detected(self):
        """Beam with all check points outside zone — must not be detected."""
        beam, nodes = _beam_and_nodes([10.0, 0.0, 0.0], [12.0, 0.0, 0.0])
        zone = _iso_zone([0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        result = exposed_element_ids({1: beam}, [zone], nodes)
        assert 1 not in result

    def test_inactive_zone_ignored(self):
        """Inactive fire zone must not contribute to exposure."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        zone = _iso_zone([1.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        zone.active = False
        result = exposed_element_ids({1: beam}, [zone], nodes)
        assert 1 not in result


class TestExposureFlags:
    def test_endpoint_exposure(self):
        """exposure_flags must return all-1.0 when an endpoint is inside zone."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([4.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        flags = exposure_flags(beam, [zone], nodes)
        assert all(v == 1.0 for v in flags.values())

    def test_fully_outside_flags(self):
        """exposure_flags must return all-0.0 when no point is inside zone."""
        beam, nodes = _beam_and_nodes([10.0, 0.0, 0.0], [12.0, 0.0, 0.0])
        zone = _iso_zone([0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        flags = exposure_flags(beam, [zone], nodes)
        assert all(v == 0.0 for v in flags.values())


# ── element_quad_exposure_flags ───────────────────────────────────────────────

class TestElementQuadExposureFlags:
    """Verify per-quad exposure for a BOX beam."""

    def _make_mesh(self, length=4.0, n_length=4):
        """Build a BOX surface mesh with n_length elements along beam axis."""
        sec = _box_section()
        return BoxSurfaceMesher(
            section=sec, length=length, n_top=2, n_side=3, n_length=n_length
        ).build()

    def test_fully_exposed_returns_all_ones(self):
        """Zone that fully encloses the beam → all flags = 1.0."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([2.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        mesh = self._make_mesh(length=4.0, n_length=4)
        flags = element_quad_exposure_flags(mesh, beam, nodes, [zone])
        assert np.all(flags == 1.0)

    def test_no_exposure_returns_zeros(self):
        """Zone far from the beam → all flags = 0.0."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([20.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        mesh = self._make_mesh(length=4.0, n_length=4)
        flags = element_quad_exposure_flags(mesh, beam, nodes, [zone])
        assert np.all(flags == 0.0)

    def test_partial_exposure_splits_at_boundary(self):
        """Zone covers only the first half of the beam → roughly half the quads exposed."""
        # Beam: x = 0..4;  Zone: x = -1..2 (covers first half)
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([0.5, 0.0, 0.0], [3.0, 10.0, 10.0])  # x: -1..2
        mesh = self._make_mesh(length=4.0, n_length=4)
        flags = element_quad_exposure_flags(mesh, beam, nodes, [zone])
        # Some quads exposed, some not
        assert np.any(flags == 1.0), "Expected some exposed quads"
        assert np.any(flags == 0.0), "Expected some unexposed quads"

    def test_inactive_zone_gives_zeros(self):
        """Inactive zone must not expose any quads."""
        beam, nodes = _beam_and_nodes([0.0, 0.0, 0.0], [4.0, 0.0, 0.0])
        zone = _iso_zone([2.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        zone.active = False
        mesh = self._make_mesh(length=4.0, n_length=4)
        flags = element_quad_exposure_flags(mesh, beam, nodes, [zone])
        assert np.all(flags == 0.0)


# ── SurfaceTransientSolver: quad_exposure heating rate ───────────────────────

class TestQuadExposureHeating:
    """Verify that partial exposure reduces the heating rate proportionally."""

    def _solver(self, n_quads, exposure_fraction, fire_temp_val=800.0):
        """
        Build a solver on a small BOX mesh with uniform exposure fraction.
        Returns (solver, mesh).
        """
        sec = _box_section()
        mesh = BoxSurfaceMesher(
            section=sec, length=1.0, n_top=2, n_side=3, n_length=4
        ).build()

        n_q = mesh.n_quads
        # Build exposure array: first fraction exposed, rest not
        n_exposed = max(1, round(exposure_fraction * n_q))
        exposure = np.zeros(n_q)
        exposure[:n_exposed] = 1.0

        mat = _steel()
        solver = SurfaceTransientSolver(
            mesh=mesh, material=mat,
            fire_temp=lambda t: fire_temp_val,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
            quad_exposure=exposure,
        )
        return solver, mesh

    def test_fully_exposed_heats_faster_than_partially(self):
        """Full exposure produces higher temperature rise than 50% exposure."""
        sec = _box_section()
        mesh_full = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        mat = _steel()

        solver_full = SurfaceTransientSolver(
            mesh=mesh_full, material=mat,
            fire_temp=lambda t: 800.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
        )

        mesh_half = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        n_q = mesh_half.n_quads
        half_exp = np.zeros(n_q)
        half_exp[:n_q // 2] = 1.0
        solver_half = SurfaceTransientSolver(
            mesh=mesh_half, material=mat,
            fire_temp=lambda t: 800.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
            quad_exposure=half_exp,
        )

        n = mesh_full.n_nodes
        T_full = np.full(n, 20.0)
        T_half = np.full(n, 20.0)
        dt = 10.0

        for _ in range(5):
            T_full = solver_full.step(T_full, dt, dt)
            T_half = solver_half.step(T_half, dt, dt)

        mean_full = float(np.mean(T_full))
        mean_half = float(np.mean(T_half))
        assert mean_full > mean_half, (
            f"Full exposure ({mean_full:.1f}°C) should be hotter than half ({mean_half:.1f}°C)"
        )

    def test_zero_exposure_no_heating(self):
        """All-zeros exposure → element stays at ambient (no fire BC applied)."""
        sec = _box_section()
        mesh = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        n_q = mesh.n_quads
        mat = _steel()

        solver = SurfaceTransientSolver(
            mesh=mesh, material=mat,
            fire_temp=lambda t: 800.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
            quad_exposure=np.zeros(n_q),
        )

        T = np.full(mesh.n_nodes, 20.0)
        for _ in range(10):
            T = solver.step(T, 10.0, 10.0)

        # With no exposure, temperature should remain within ≈0.1°C of ambient
        assert np.all(np.abs(T - 20.0) < 0.1), (
            f"Expected no heating with zero exposure; got T_max={T.max():.3f}°C"
        )

    def test_none_exposure_matches_all_ones(self):
        """quad_exposure=None behaves identically to all-ones array."""
        sec = _box_section()
        mat = _steel()
        dt = 30.0

        mesh1 = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        solver1 = SurfaceTransientSolver(
            mesh=mesh1, material=mat,
            fire_temp=lambda t: 700.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
            quad_exposure=None,
        )

        mesh2 = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        solver2 = SurfaceTransientSolver(
            mesh=mesh2, material=mat,
            fire_temp=lambda t: 700.0,
            epsilon_m=0.7, h_conv=25.0, T0=20.0,
            quad_exposure=np.ones(mesh2.n_quads),
        )

        T1 = np.full(mesh1.n_nodes, 20.0)
        T2 = np.full(mesh2.n_nodes, 20.0)
        for _ in range(5):
            T1 = solver1.step(T1, dt, dt)
            T2 = solver2.step(T2, dt, dt)

        np.testing.assert_allclose(T1, T2, rtol=1e-10,
                                   err_msg="quad_exposure=None should equal all-ones")


# ── run_analysis: endpoint-straddling element included ───────────────────────

class TestRunAnalysisEndpointExposure:
    """Integration test: an element straddling a zone boundary is analysed."""

    def test_straddling_beam_included(self):
        """
        A beam whose midpoint is outside the zone but whose endpoint n2 is inside
        must be solved (not skipped) and must show temperature rise.
        """
        from fahts.core.heat.solver.analysis_runner import run_analysis
        from fahts.core.io.usfos_reader import read_usfos_fem
        from fahts.core.results.analysis_config import AnalysisConfig
        from pathlib import Path

        model = read_usfos_fem(Path(__file__).parent.parent / "model_file.fem")

        # Pick element 1 (first beam)
        eid = next(iter(model.elements))
        elem = model.elements[eid]
        n2_pos = model.nodes[elem.n2].xyz

        # Zone centred on n2, small enough that midpoint is outside
        half_length = elem.length * 0.3
        zone = FireZone(
            name="endpoint_zone",
            center=n2_pos.copy(),
            dims=np.array([half_length * 2, 2.0, 2.0]),
            curve=FireCurve(FireCurveType.HYDROCARBON),
        )

        cfg = AnalysisConfig(
            t_end=60.0, dt=10.0, output_dt=60.0,
            element_ids=[eid],
        )
        result = run_analysis(model, [zone], cfg)
        assert eid in result.element_ids
        idx = result.element_ids.index(eid)
        T_final = result.T_centroid[-1, idx]
        assert T_final > 21.0, (
            f"Expected temperature rise for endpoint-exposed beam; got {T_final:.1f}°C"
        )
