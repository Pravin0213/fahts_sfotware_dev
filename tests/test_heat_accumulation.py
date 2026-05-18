"""
Tests for Phase 3F Step 2 — heat accumulation element for hollow BOX/PIPE sections.

Covers:
  - SectionMesh.inner_node_indices property
  - TransientSolver M_extra parameter
  - analysis_runner._compute_M_extra helper
  - End-to-end: BOX and PIPE inner nodes heat more slowly with air mass
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.section_mesh.box_mesher import BoxMesher
from fahts.core.heat.section_mesh.ihprofil_mesher import IProfileMesher
from fahts.core.heat.section_mesh.pipe_mesher import PipeMesher
from fahts.core.heat.solver.analysis_runner import _compute_M_extra, run_analysis
from fahts.core.heat.solver.time_integrator import TransientSolver
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, ISection, PipeSection
from fahts.core.results.analysis_config import AnalysisConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _box_sec() -> BoxSection:
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def _pipe_sec() -> PipeSection:
    return PipeSection(sid=2, outer_diameter=0.3, thickness=0.02)


def _isec() -> ISection:
    return ISection(sid=3, h=0.30, tw=0.010, bf_top=0.15, tf_top=0.012,
                    bf_bot=0.15, tf_bot=0.012)


def _mat() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


# ── SectionMesh.inner_node_indices ────────────────────────────────────────────

class TestInnerNodeIndices:
    def test_box_has_inner_nodes(self):
        mesh = BoxMesher(_box_sec()).build()
        ids = mesh.inner_node_indices
        assert len(ids) > 0

    def test_box_inner_nodes_valid_indices(self):
        mesh = BoxMesher(_box_sec()).build()
        for idx in mesh.inner_node_indices:
            assert 0 <= idx < mesh.n_nodes

    def test_box_inner_nodes_unique(self):
        mesh = BoxMesher(_box_sec()).build()
        ids = mesh.inner_node_indices
        assert len(ids) == len(set(ids))

    def test_pipe_has_inner_nodes(self):
        mesh = PipeMesher(_pipe_sec()).build()
        ids = mesh.inner_node_indices
        assert len(ids) > 0

    def test_pipe_inner_nodes_valid_indices(self):
        mesh = PipeMesher(_pipe_sec()).build()
        for idx in mesh.inner_node_indices:
            assert 0 <= idx < mesh.n_nodes

    def test_isection_no_inner_nodes(self):
        """I/H profiles are solid — no enclosed air cavity."""
        mesh = IProfileMesher(_isec()).build()
        assert mesh.inner_node_indices == []

    def test_inner_node_indices_sorted(self):
        """Result should be a sorted list for determinism."""
        mesh = BoxMesher(_box_sec()).build()
        ids = mesh.inner_node_indices
        assert ids == sorted(ids)

    def test_inner_nodes_are_subset_of_inner_edge_pairs(self):
        mesh = BoxMesher(_box_sec()).build()
        pair_nodes = {int(n) for pair in mesh.inner_edge_pairs for n in pair}
        assert set(mesh.inner_node_indices) == pair_nodes


# ── _compute_M_extra helper ───────────────────────────────────────────────────

class TestComputeMExtra:
    def test_box_returns_array(self):
        sec = _box_sec()
        mesh = BoxMesher(sec).build()
        result = _compute_M_extra(sec, mesh, elem_length=1.0)
        assert result is not None
        assert result.shape == (mesh.n_nodes,)

    def test_box_nonzero_only_at_inner_nodes(self):
        sec = _box_sec()
        mesh = BoxMesher(sec).build()
        M_extra = _compute_M_extra(sec, mesh, elem_length=1.0)
        inner_ids = set(mesh.inner_node_indices)
        for idx in range(mesh.n_nodes):
            if idx in inner_ids:
                assert M_extra[idx] > 0.0
            else:
                assert M_extra[idx] == 0.0

    def test_box_total_mass_correct(self):
        """Total extra mass should equal A_inner * L * 1200."""
        sec = _box_sec()
        mesh = BoxMesher(sec).build()
        L = 2.5
        M_extra = _compute_M_extra(sec, mesh, elem_length=L)
        expected_total = sec.inner_height * sec.inner_width * L * 1200.0
        assert abs(M_extra.sum() - expected_total) < 1e-10

    def test_pipe_returns_array(self):
        sec = _pipe_sec()
        mesh = PipeMesher(sec).build()
        result = _compute_M_extra(sec, mesh, elem_length=1.0)
        assert result is not None
        assert result.shape == (mesh.n_nodes,)

    def test_pipe_total_mass_correct(self):
        sec = _pipe_sec()
        mesh = PipeMesher(sec).build()
        L = 3.0
        M_extra = _compute_M_extra(sec, mesh, elem_length=L)
        expected_total = math.pi * sec.inner_radius ** 2 * L * 1200.0
        assert abs(M_extra.sum() - expected_total) < 1e-10

    def test_isection_returns_none(self):
        """Solid I/H profiles have no enclosed air — must return None."""
        sec = _isec()
        mesh = IProfileMesher(sec).build()
        result = _compute_M_extra(sec, mesh, elem_length=1.0)
        assert result is None

    def test_longer_element_gives_larger_mass(self):
        sec = _box_sec()
        mesh = BoxMesher(sec).build()
        M1 = _compute_M_extra(sec, mesh, elem_length=1.0)
        M2 = _compute_M_extra(sec, mesh, elem_length=2.0)
        assert M2.sum() == pytest.approx(2.0 * M1.sum())


# ── TransientSolver with M_extra ─────────────────────────────────────────────

class TestTransientSolverMExtra:
    def _run(self, M_extra=None) -> np.ndarray:
        """Run 300 s with constant 900°C fire, return final temperature array."""
        sec = _box_sec()
        mesh = BoxMesher(sec, n_layers=1).build()
        solver = TransientSolver(
            mesh=mesh,
            material=_mat(),
            fire_temp=lambda t: 900.0,
            epsilon_m=0.7,
            h_conv=25.0,
            M_extra=M_extra,
        )
        _, T_hist = solver.run(t_end=300.0, dt=30.0, output_dt=300.0)
        return T_hist[-1]

    def test_no_M_extra_accepted(self):
        """M_extra=None must not raise — existing code path unchanged."""
        T = self._run(M_extra=None)
        assert T.shape[0] > 0

    def test_M_extra_slows_inner_temperature(self):
        """With air mass, inner nodes should be cooler than without."""
        sec = _box_sec()
        mesh = BoxMesher(sec, n_layers=1).build()
        inner_ids = mesh.inner_node_indices

        L = 1.0
        M_extra = _compute_M_extra(sec, mesh, elem_length=L)

        T_bare = self._run(M_extra=None)
        T_acc  = self._run(M_extra=M_extra)

        assert np.mean(T_acc[inner_ids]) < np.mean(T_bare[inner_ids]), (
            "Inner nodes should be cooler when air heat accumulation is active"
        )

    def test_M_extra_does_not_affect_outer_temperature_much(self):
        """
        Outer nodes are directly exposed to fire; the air mass (on inner nodes)
        should not significantly change outer node temperatures.
        The outer node mean should differ by less than 10°C after 300s.
        """
        sec = _box_sec()
        mesh = BoxMesher(sec, n_layers=1).build()
        outer_ids = np.unique(mesh.outer_edge_pairs)
        M_extra = _compute_M_extra(sec, mesh, elem_length=1.0)

        T_bare = self._run(M_extra=None)
        T_acc  = self._run(M_extra=M_extra)

        diff = abs(np.mean(T_bare[outer_ids]) - np.mean(T_acc[outer_ids]))
        assert diff < 10.0, f"Outer temperature shifted by {diff:.1f}°C — too much"


# ── End-to-end: run_analysis with BOX element ─────────────────────────────────

class TestRunAnalysisHeatAccumulation:
    """Integration test: heat accumulation is applied automatically for BOX/PIPE."""

    def _make_model(self, sec) -> FEMModel:
        n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(nid=2, x=2.0, y=0.0, z=0.0)
        mat = _mat()
        elem = BeamElement(
            eid=1, n1=1, n2=2, mat_id=1, geom_id=sec.sid, lcoor_id=0,
            length=2.0,
            direction=np.array([1.0, 0.0, 0.0]),
            local_z=np.array([0.0, 0.0, 1.0]),
        )
        return FEMModel(
            nodes={1: n1, 2: n2},
            elements={1: elem},
            sections={sec.sid: sec},
            materials={1: mat},
            groups={}, unitvecs={},
            source_file=Path("synthetic.fem"),
        )

    def _make_zone(self) -> FireZone:
        return FireZone(
            name="Z",
            center=np.array([1.0, 0.0, 0.0]),
            dims=np.array([4.0, 4.0, 4.0]),
            curve=FireCurve(FireCurveType.ISO_834),
            epsilon_fire=1.0, h_conv=25.0, active=True,
        )

    def _cfg(self) -> AnalysisConfig:
        return AnalysisConfig(t_end=300.0, dt=30.0, output_dt=300.0, n_layers=1)

    def test_box_analysis_completes(self):
        result = run_analysis(self._make_model(_box_sec()), [self._make_zone()], self._cfg())
        assert result.element_ids == [1]

    def test_pipe_analysis_completes(self):
        result = run_analysis(self._make_model(_pipe_sec()), [self._make_zone()], self._cfg())
        assert result.element_ids == [1]

    def test_box_temperature_increases(self):
        result = run_analysis(self._make_model(_box_sec()), [self._make_zone()], self._cfg())
        T_cen = result.T_centroid[:, 0]
        assert T_cen[-1] > T_cen[0]
