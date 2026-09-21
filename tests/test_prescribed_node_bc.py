"""
Tests for §3.5.2 prescribed nodal boundary temperature.

Covers:
  - PrescribedNodeBC dataclass construction and validation
  - eval() for constant and callable temperatures
  - SurfaceTransientSolver: prescribed node stays exactly at target every step
  - SurfaceTransientSolver: unprescribed neighbour responds to gradient
  - SurfaceTransientSolver: callable (time-varying) temperature prescription
  - SurfaceTransientSolver: multiple independent PrescribedNodeBC entries
  - AnalysisConfig.prescribed_node_bcs field exists and defaults to empty list
  - Wire-through: prescribed_node_bcs forwarded from AnalysisConfig through run_analysis
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection
from fahts.core.results.analysis_config import AnalysisConfig


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _box_sec() -> BoxSection:
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def _steel() -> SteelMaterial:
    return SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=355e6, rho=7850.0, alpha_T=1.2e-5)


def _mesh() -> BeamSurfaceMesh:
    return BoxSurfaceMesher(section=_box_sec(), length=1.0,
                            n_top=2, n_side=2, n_length=2).build()


def _fire_800(t: float) -> float:
    return 800.0


def _base_solver(mesh: BeamSurfaceMesh, **kw) -> SurfaceTransientSolver:
    return SurfaceTransientSolver(
        mesh=mesh,
        material=_steel(),
        fire_temp=_fire_800,
        epsilon_m=0.5,
        h_conv=25.0,
        T0=20.0,
        **kw,
    )


# ── PrescribedNodeBC dataclass ────────────────────────────────────────────────

class TestPrescribedNodeBCDataclass:
    """Unit tests for PrescribedNodeBC construction and eval()."""

    def test_constant_float_construction(self):
        bc = PrescribedNodeBC(node_indices=[0, 1], temperature=300.0)
        assert bc.node_indices == [0, 1]
        assert bc.temperature == 300.0

    def test_callable_construction(self):
        fn = lambda t: 20.0 + 5.0 * t
        bc = PrescribedNodeBC(node_indices=[2], temperature=fn)
        assert callable(bc.temperature)

    def test_eval_constant(self):
        bc = PrescribedNodeBC(node_indices=[0], temperature=500.0)
        assert bc.eval(0.0) == pytest.approx(500.0)
        assert bc.eval(100.0) == pytest.approx(500.0)
        assert bc.eval(3600.0) == pytest.approx(500.0)

    def test_eval_callable(self):
        bc = PrescribedNodeBC(node_indices=[0], temperature=lambda t: 20.0 + 2.0 * t)
        assert bc.eval(0.0) == pytest.approx(20.0)
        assert bc.eval(10.0) == pytest.approx(40.0)
        assert bc.eval(100.0) == pytest.approx(220.0)

    def test_empty_node_indices_raises(self):
        with pytest.raises(ValueError, match="node_indices must not be empty"):
            PrescribedNodeBC(node_indices=[], temperature=100.0)

    def test_bad_temperature_type_raises(self):
        with pytest.raises(TypeError, match="temperature must be a float or a callable"):
            PrescribedNodeBC(node_indices=[0], temperature="hot")  # type: ignore[arg-type]

    def test_integer_temperature_accepted(self):
        """int should be accepted as a numeric temperature."""
        bc = PrescribedNodeBC(node_indices=[0], temperature=400)
        assert bc.eval(0.0) == pytest.approx(400.0)

    def test_out_of_range_index_raises_in_solver(self):
        """Solver should reject PrescribedNodeBC with an out-of-range node index."""
        mesh = _mesh()
        bc = PrescribedNodeBC(node_indices=[mesh.n_nodes + 100], temperature=300.0)
        with pytest.raises(ValueError, match="out of range"):
            _base_solver(mesh, prescribed_node_bcs=[bc])


# ── SurfaceTransientSolver: constant prescribed temperature ───────────────────

class TestPrescribedNodeSolverConstant:
    """A prescribed node must stay exactly at the target temperature every step."""

    def test_prescribed_node_exact_at_every_step(self):
        """Node 0 pinned to 400 °C: must be exactly 400 °C at every output step."""
        mesh    = _mesh()
        T_presc = 400.0
        bc      = PrescribedNodeBC(node_indices=[0], temperature=T_presc)
        solver  = _base_solver(mesh, prescribed_node_bcs=[bc])

        _, T_hist = solver.run(t_end=300.0, dt=30.0, output_dt=60.0)
        # T_hist shape: (n_out, n_nodes); check node 0 at all output times (skip t=0)
        for step_idx in range(1, T_hist.shape[0]):
            assert T_hist[step_idx, 0] == pytest.approx(T_presc, abs=1e-10), (
                f"Prescribed node 0 deviated at output step {step_idx}: "
                f"{T_hist[step_idx, 0]:.6f} != {T_presc}"
            )

    def test_prescribed_node_at_initial_temp(self):
        """Node pinned to 20 °C (initial): should stay at 20 °C while others heat."""
        mesh    = _mesh()
        bc      = PrescribedNodeBC(node_indices=[0], temperature=20.0)
        solver  = _base_solver(mesh, prescribed_node_bcs=[bc])

        _, T_hist = solver.run(t_end=120.0, dt=20.0, output_dt=60.0)
        # Prescribed node stays at 20 °C
        for step_idx in range(1, T_hist.shape[0]):
            assert T_hist[step_idx, 0] == pytest.approx(20.0, abs=1e-10)
        # Non-prescribed nodes heat above 20 °C
        assert float(np.max(T_hist[-1, 1:])) > 20.0

    def test_multiple_prescribed_nodes(self):
        """Multiple nodes can be listed in one PrescribedNodeBC entry."""
        mesh    = _mesh()
        bc      = PrescribedNodeBC(node_indices=[0, 1, 2], temperature=350.0)
        solver  = _base_solver(mesh, prescribed_node_bcs=[bc])

        _, T_hist = solver.run(t_end=120.0, dt=30.0, output_dt=60.0)
        for i in [0, 1, 2]:
            assert T_hist[-1, i] == pytest.approx(350.0, abs=1e-10), (
                f"Prescribed node {i} deviated: {T_hist[-1, i]:.6f}"
            )

    def test_unprescribed_nodes_respond_to_fire(self):
        """Unprescribed nodes must still heat under fire exposure."""
        mesh    = _mesh()
        bc      = PrescribedNodeBC(node_indices=[0], temperature=20.0)
        solver  = _base_solver(mesh, prescribed_node_bcs=[bc])

        _, T_hist = solver.run(t_end=300.0, dt=30.0, output_dt=300.0)
        # Unprescribed nodes should be well above 20 °C by t=300 s
        unprescribed_T = T_hist[-1, 1:]
        assert float(np.min(unprescribed_T)) > 25.0, (
            "Unprescribed nodes should heat under fire exposure."
        )

    def test_two_independent_bcs_both_enforced(self):
        """Two separate PrescribedNodeBC entries both get enforced simultaneously."""
        mesh = _mesh()
        bc0  = PrescribedNodeBC(node_indices=[0], temperature=300.0)
        bc1  = PrescribedNodeBC(node_indices=[1], temperature=500.0)
        solver = _base_solver(mesh, prescribed_node_bcs=[bc0, bc1])

        _, T_hist = solver.run(t_end=120.0, dt=30.0, output_dt=60.0)
        assert T_hist[-1, 0] == pytest.approx(300.0, abs=1e-10)
        assert T_hist[-1, 1] == pytest.approx(500.0, abs=1e-10)


# ── SurfaceTransientSolver: callable (time-varying) temperature ───────────────

class TestPrescribedNodeSolverCallable:
    """Callable (time-varying) temperature prescriptions must be evaluated per step."""

    def test_callable_temperature_tracked(self):
        """Node 0 prescribed to ramp: T(t) = 20 + t [°C/s]."""
        mesh = _mesh()
        bc   = PrescribedNodeBC(node_indices=[0], temperature=lambda t: 20.0 + t)
        solver = _base_solver(mesh, prescribed_node_bcs=[bc])

        out_times, T_hist = solver.run(t_end=100.0, dt=10.0, output_dt=10.0)
        for idx, t_out in enumerate(out_times):
            expected = 20.0 + t_out
            actual   = T_hist[idx, 0]
            assert actual == pytest.approx(expected, abs=1e-10), (
                f"Callable-prescribed node 0 at t={t_out:.1f} s: "
                f"expected {expected:.4f}, got {actual:.6f}"
            )

    def test_constant_and_callable_coexist(self):
        """One constant and one callable BC can coexist without interference."""
        mesh = _mesh()
        bc0  = PrescribedNodeBC(node_indices=[0], temperature=300.0)
        bc1  = PrescribedNodeBC(node_indices=[1], temperature=lambda t: 20.0 + 2.0 * t)
        solver = _base_solver(mesh, prescribed_node_bcs=[bc0, bc1])

        out_times, T_hist = solver.run(t_end=60.0, dt=10.0, output_dt=10.0)
        for idx, t_out in enumerate(out_times):
            assert T_hist[idx, 0] == pytest.approx(300.0, abs=1e-10)
            assert T_hist[idx, 1] == pytest.approx(20.0 + 2.0 * t_out, abs=1e-10)


# ── Gradient driven by prescribed BC ─────────────────────────────────────────

class TestPrescribedNodeGradient:
    """A hot prescribed BC should drive a temperature gradient in the mesh."""

    def test_hot_prescribed_node_drives_gradient(self):
        """
        Node 0 prescribed to 600 °C while the rest start at 20 °C.
        Fire is off (epsilon_m=0, h_conv=0, fire_temp=20 °C).
        Conduction through the mesh should heat the neighbours above 20 °C.
        """
        mesh = _mesh()
        bc   = PrescribedNodeBC(node_indices=[0], temperature=600.0)

        solver = SurfaceTransientSolver(
            mesh=mesh,
            material=_steel(),
            fire_temp=lambda t: 20.0,  # no fire
            epsilon_m=0.0,
            h_conv=0.0,
            T0=20.0,
            prescribed_node_bcs=[bc],
        )

        _, T_hist = solver.run(t_end=600.0, dt=60.0, output_dt=600.0)
        # Node 0 must be exactly 600 °C
        assert T_hist[-1, 0] == pytest.approx(600.0, abs=1e-10)
        # At least one unprescribed neighbour must have been driven above 20 °C
        assert float(np.max(T_hist[-1, 1:])) > 20.0, (
            "Conduction from prescribed hot node should heat neighbours."
        )


# ── AnalysisConfig field ──────────────────────────────────────────────────────

class TestAnalysisConfigField:
    """AnalysisConfig.prescribed_node_bcs field must exist and default to []."""

    def test_default_empty_list(self):
        cfg = AnalysisConfig(t_end=60.0, dt=10.0, output_dt=60.0)
        assert hasattr(cfg, "prescribed_node_bcs")
        assert cfg.prescribed_node_bcs == []

    def test_can_set_prescribed_node_bcs(self):
        bc  = PrescribedNodeBC(node_indices=[0], temperature=300.0)
        cfg = AnalysisConfig(
            t_end=60.0, dt=10.0, output_dt=60.0,
            prescribed_node_bcs=[bc],
        )
        assert len(cfg.prescribed_node_bcs) == 1
        assert cfg.prescribed_node_bcs[0] is bc

    def test_validate_does_not_raise_for_valid_bcs(self):
        bc  = PrescribedNodeBC(node_indices=[0, 1], temperature=lambda t: t)
        cfg = AnalysisConfig(
            t_end=60.0, dt=10.0, output_dt=60.0,
            prescribed_node_bcs=[bc],
        )
        cfg.validate()  # should not raise


# ── Wire-through test via run_analysis ────────────────────────────────────────

class TestWireThroughRunAnalysis:
    """
    prescribed_node_bcs in AnalysisConfig must be forwarded into the per-element
    SurfaceTransientSolver by run_analysis.
    """

    def test_prescribed_node_fixed_in_full_analysis(self):
        """
        Run a minimal analysis on a single-element BOX model.
        Node 0 prescribed to 400 °C → must be exactly 400 °C in the result.
        Uses the same model-building pattern as test_analysis_runner.py.
        """
        from fahts.core.heat.solver.analysis_runner import run_analysis
        from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
        from fahts.core.model.element import BeamElement
        from fahts.core.model.fem_model import FEMModel
        from fahts.core.model.node import Node

        # Beam runs from (0,0,0) to (1,0,0); fire zone covers midpoint (0.5,0,0)
        n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(nid=2, x=1.0, y=0.0, z=0.0)
        nodes = {1: n1, 2: n2}

        mat  = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)
        sec  = BoxSection(sid=10, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
        elem = BeamElement(
            eid=101, n1=1, n2=2, mat_id=1, geom_id=10, lcoor_id=0,
            length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            local_z=np.array([0.0, 0.0, 1.0]),
        )

        model = FEMModel(
            nodes=nodes,
            elements={101: elem},
            sections={10: sec},
            materials={1: mat},
            groups={},
            unitvecs={},
            source_file=Path("synthetic_presc.fem"),
        )

        # Fire zone covering the element midpoint
        fire_zone = FireZone(
            name="test_zone",
            center=np.array([0.5, 0.0, 0.0]),
            dims=np.array([2.0, 2.0, 2.0]),
            curve=FireCurve(curve_type=FireCurveType.ISO_834),
            h_conv=25.0,
            epsilon_fire=1.0,
            active=True,
        )

        # Prescribe node 0 of the element's surface mesh to 400 °C
        bc  = PrescribedNodeBC(node_indices=[0], temperature=400.0)
        cfg = AnalysisConfig(
            t_end=60.0, dt=30.0, output_dt=60.0,
            element_ids=[101],
            n_top=1, n_side=1, n_length=1,
            prescribed_node_bcs=[bc],
        )

        result = run_analysis(model=model, fire_zones=[fire_zone], config=cfg)
        # T_section[eid] shape: (n_steps, n_nodes); check final step node 0
        T_section_last = result.T_section[101][-1]   # (n_nodes,)
        assert T_section_last[0] == pytest.approx(400.0, abs=1e-8), (
            f"Wire-through test: node 0 should be 400 °C, got {T_section_last[0]:.6f}"
        )
