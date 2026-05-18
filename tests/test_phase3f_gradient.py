"""
Tests for Phase 3F Step 3 — temperature gradient (βy, βz) computation.

Covers:
  - TemperatureField.section_gradient()
  - export_beltemp() with and without the meshes parameter
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.io.results_writer import export_beltemp
from fahts.core.results.temperature_field import TemperatureField


# ── Helpers ───────────────────────────────────────────────────────────────────

def _two_quad_mesh() -> SectionMesh:
    """
    Minimal 2-quad SectionMesh centred on origin.

    Layout (y horizontal, z vertical):
        y: -1 ——— 0 ——— +1
        z: -0.5 to +0.5 (height = 1)

    Left quad  (nodes 0,1,2,3): centroid y=-0.5, z=0, area=1.0
    Right quad (nodes 1,4,5,2): centroid y=+0.5, z=0, area=1.0
    """
    nodes = np.array([
        [-1.0, -0.5],  # 0
        [ 0.0, -0.5],  # 1
        [ 0.0,  0.5],  # 2
        [-1.0,  0.5],  # 3
        [ 1.0, -0.5],  # 4
        [ 1.0,  0.5],  # 5
    ])
    quads = np.array([
        [0, 1, 2, 3],   # left
        [1, 4, 5, 2],   # right
    ])
    empty = np.empty((0, 2), dtype=int)
    return SectionMesh(nodes=nodes, quads=quads,
                       outer_edge_pairs=empty, inner_edge_pairs=empty)


def _two_quad_mesh_tall() -> SectionMesh:
    """
    2-quad SectionMesh with quads stacked vertically (non-zero z-centroids).

    Bottom quad centroid y=0, z=-0.5, area=1.0
    Top quad    centroid y=0, z=+0.5, area=1.0
    """
    nodes = np.array([
        [-0.5, -1.0],  # 0
        [ 0.5, -1.0],  # 1
        [ 0.5,  0.0],  # 2
        [-0.5,  0.0],  # 3
        [-0.5,  1.0],  # 4
        [ 0.5,  1.0],  # 5
    ])
    quads = np.array([
        [0, 1, 2, 3],   # bottom
        [3, 2, 5, 4],   # top
    ])
    empty = np.empty((0, 2), dtype=int)
    return SectionMesh(nodes=nodes, quads=quads,
                       outer_edge_pairs=empty, inner_edge_pairs=empty)


def _make_tf(
    eid: int,
    n_steps: int,
    T_nodal: np.ndarray,
    mesh_n_nodes: int,
) -> TemperatureField:
    """
    Build a single-element TemperatureField with constant nodal temperature history.
    T_nodal: (mesh_n_nodes,) — same values at every time step.
    """
    times = np.arange(n_steps, dtype=float)
    T_section_hist = np.tile(T_nodal, (n_steps, 1))   # (n_steps, n_nodes)
    T_centroid = np.full((n_steps, 1), T_nodal.mean())
    return TemperatureField(
        times=times,
        element_ids=[eid],
        T_centroid=T_centroid,
        T_section={eid: T_section_hist},
    )


# ── section_gradient — uniform temperature ────────────────────────────────────

class TestSectionGradientUniform:
    """Uniform temperature → both gradients must be zero."""

    def test_beta_y_is_zero(self):
        mesh = _two_quad_mesh()
        T = np.full(6, 300.0)
        tf = _make_tf(eid=1, n_steps=3, T_nodal=T, mesh_n_nodes=6)
        beta_y, _ = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_y) < 1e-10

    def test_beta_z_is_zero(self):
        mesh = _two_quad_mesh()
        T = np.full(6, 300.0)
        tf = _make_tf(eid=1, n_steps=3, T_nodal=T, mesh_n_nodes=6)
        _, beta_z = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_z) < 1e-10

    def test_last_step_default_index(self):
        mesh = _two_quad_mesh()
        T = np.full(6, 200.0)
        tf = _make_tf(eid=1, n_steps=5, T_nodal=T, mesh_n_nodes=6)
        by, bz = tf.section_gradient(1, t_idx=-1, mesh=mesh)
        assert abs(by) < 1e-10
        assert abs(bz) < 1e-10


# ── section_gradient — linear gradient in y (βz) ─────────────────────────────

class TestSectionGradientLinearY:
    """T = c·y  →  βz = c exactly; βy ≈ 0."""

    @pytest.mark.parametrize("c", [100.0, -50.0, 0.5])
    def test_beta_z_equals_coefficient(self, c: float):
        """
        Two-quad mesh with quads at y=-0.5 and y=+0.5.
        Nodal temperatures: T_i = c * y_i.

        βz = Σ(T_k · y_k · A_k) / Iz
           = [c·(-0.5)·(-0.5)·1 + c·0.5·0.5·1] / [(-0.5)²·1 + 0.5²·1]
           = c·0.5 / 0.5 = c
        """
        mesh = _two_quad_mesh()
        y_coords = mesh.nodes[:, 0]
        T = c * y_coords
        tf = _make_tf(eid=1, n_steps=2, T_nodal=T, mesh_n_nodes=6)
        _, beta_z = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_z - c) < 1e-9

    @pytest.mark.parametrize("c", [100.0, -50.0])
    def test_beta_y_near_zero_for_y_gradient(self, c: float):
        mesh = _two_quad_mesh()
        y_coords = mesh.nodes[:, 0]
        T = c * y_coords
        tf = _make_tf(eid=1, n_steps=2, T_nodal=T, mesh_n_nodes=6)
        beta_y, _ = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_y) < 1e-9


# ── section_gradient — linear gradient in z (βy) ─────────────────────────────

class TestSectionGradientLinearZ:
    """T = c·z  →  βy = c exactly; βz ≈ 0."""

    @pytest.mark.parametrize("c", [80.0, -30.0, 1.0])
    def test_beta_y_equals_coefficient(self, c: float):
        """
        Tall two-quad mesh with quads at z=-0.5 and z=+0.5.
        Nodal temperatures: T_i = c * z_i  →  βy = c.
        """
        mesh = _two_quad_mesh_tall()
        z_coords = mesh.nodes[:, 1]
        T = c * z_coords
        tf = _make_tf(eid=1, n_steps=2, T_nodal=T, mesh_n_nodes=6)
        beta_y, _ = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_y - c) < 1e-9

    @pytest.mark.parametrize("c", [80.0, -30.0])
    def test_beta_z_near_zero_for_z_gradient(self, c: float):
        mesh = _two_quad_mesh_tall()
        z_coords = mesh.nodes[:, 1]
        T = c * z_coords
        tf = _make_tf(eid=1, n_steps=2, T_nodal=T, mesh_n_nodes=6)
        _, beta_z = tf.section_gradient(1, t_idx=0, mesh=mesh)
        assert abs(beta_z) < 1e-9


# ── section_gradient — missing section data ───────────────────────────────────

class TestSectionGradientNoData:
    def test_returns_zero_when_no_T_section(self):
        mesh = _two_quad_mesh()
        # TemperatureField with no T_section entry for eid=42
        tf = TemperatureField(
            times=np.array([0.0, 1.0]),
            element_ids=[42],
            T_centroid=np.ones((2, 1)) * 100.0,
        )
        beta_y, beta_z = tf.section_gradient(42, t_idx=0, mesh=mesh)
        assert beta_y == 0.0
        assert beta_z == 0.0


# ── section_gradient — with BoxMesher ────────────────────────────────────────

class TestSectionGradientBoxMesher:
    """Integration test: real BOX mesh, uniform T → zero gradients."""

    def test_uniform_temperature_zero_gradient(self):
        from fahts.core.heat.section_mesh.box_mesher import BoxMesher
        from fahts.core.model.section import BoxSection

        box = BoxSection(sid=1, H=0.2, T_side=0.01, T_bot=0.01, T_top=0.01, W=0.1)
        mesh = BoxMesher(box, n_layers=1).build()

        T = np.full(mesh.n_nodes, 400.0)
        tf = _make_tf(eid=5, n_steps=3, T_nodal=T, mesh_n_nodes=mesh.n_nodes)
        by, bz = tf.section_gradient(5, t_idx=1, mesh=mesh)
        assert abs(by) < 1e-8
        assert abs(bz) < 1e-8

    def test_linear_y_gradient_recovers_coefficient(self):
        from fahts.core.heat.section_mesh.box_mesher import BoxMesher
        from fahts.core.model.section import BoxSection

        box = BoxSection(sid=1, H=0.2, T_side=0.01, T_bot=0.01, T_top=0.01, W=0.1)
        mesh = BoxMesher(box, n_layers=1).build()

        c = 500.0  # °C/m
        T = 300.0 + c * mesh.nodes[:, 0]   # linear in y
        tf = _make_tf(eid=5, n_steps=2, T_nodal=T, mesh_n_nodes=mesh.n_nodes)
        _, beta_z = tf.section_gradient(5, t_idx=0, mesh=mesh)
        # Recovered βz should match the imposed coefficient within numerical precision
        assert abs(beta_z - c) / abs(c) < 1e-6


# ── export_beltemp — gradient columns ────────────────────────────────────────

def _make_tf_with_section(eid: int, mesh: SectionMesh) -> TemperatureField:
    """TemperatureField with a linear-in-y temperature distribution."""
    n_steps = 3
    times = np.array([0.0, 60.0, 120.0])
    c = 200.0  # °C/m gradient
    T_base = 50.0 + c * mesh.nodes[:, 0]
    T_section_hist = np.tile(T_base, (n_steps, 1))
    T_centroid = np.full((n_steps, 1), T_base.mean())
    return TemperatureField(
        times=times,
        element_ids=[eid],
        T_centroid=T_centroid,
        T_section={eid: T_section_hist},
    )


class TestExportBeltempGradient:
    def test_without_meshes_writes_zero_gradients(self, tmp_path):
        mesh = _two_quad_mesh()
        tf = _make_tf_with_section(eid=1, mesh=mesh)
        out = tmp_path / "out.fem"
        export_beltemp(tf, out)
        lines = [l for l in out.read_text().splitlines() if l.startswith(" BELTEMP")]
        # All gradient columns should be 0.000
        for line in lines:
            parts = line.split()
            assert parts[4] == "0.000"
            assert parts[5] == "0.000"

    def test_with_meshes_writes_nonzero_beta_z(self, tmp_path):
        mesh = _two_quad_mesh()
        tf = _make_tf_with_section(eid=1, mesh=mesh)
        out = tmp_path / "out_grad.fem"
        export_beltemp(tf, out, meshes={1: mesh})
        lines = [l for l in out.read_text().splitlines() if l.startswith(" BELTEMP")]
        assert lines, "No BELTEMP lines written"
        # At step 0 the gradient increment should be non-zero for βz
        # (temperature is linear in y, so βz ≠ 0)
        parts_0 = lines[0].split()
        beta_z_0 = float(parts_0[5])
        assert abs(beta_z_0) > 1e-3, "Expected non-zero βz at step 0"

    def test_with_meshes_writes_zero_beta_y_for_y_only_gradient(self, tmp_path):
        """If T is linear only in y, βy should be ~0."""
        mesh = _two_quad_mesh()
        tf = _make_tf_with_section(eid=1, mesh=mesh)
        out = tmp_path / "out_by.fem"
        export_beltemp(tf, out, meshes={1: mesh})
        lines = [l for l in out.read_text().splitlines() if l.startswith(" BELTEMP")]
        parts_0 = lines[0].split()
        beta_y_0 = float(parts_0[4])
        assert abs(beta_y_0) < 1e-3

    def test_incremental_gradients_accumulate_correctly(self, tmp_path):
        """Constant temperature history → gradient increments are zero after step 0."""
        mesh = _two_quad_mesh()
        tf = _make_tf_with_section(eid=1, mesh=mesh)
        out = tmp_path / "out_inc.fem"
        export_beltemp(tf, out, meshes={1: mesh})
        lines = [l for l in out.read_text().splitlines() if l.startswith(" BELTEMP")]
        # Constant T_section → βz same at every step → increments after step 0 are 0
        for line in lines[1:]:
            parts = line.split()
            assert abs(float(parts[5])) < 1e-6, f"Non-zero increment at step>0: {line}"

    def test_element_without_mesh_gets_zero_gradient(self, tmp_path):
        """Element not in meshes dict gets zero gradients even when other elements have meshes."""
        mesh = _two_quad_mesh()
        times = np.array([0.0, 60.0])
        T_cen = np.array([[50.0, 100.0], [60.0, 110.0]])
        tf = TemperatureField(
            times=times,
            element_ids=[1, 2],
            T_centroid=T_cen,
            T_section={
                1: np.tile(200.0 * mesh.nodes[:, 0], (2, 1)),
            },
        )
        out = tmp_path / "out_mix.fem"
        export_beltemp(tf, out, meshes={1: mesh})
        lines = [l for l in out.read_text().splitlines() if l.startswith(" BELTEMP")]
        # Element 2 should still get 0.000 gradients
        eid2_lines = [l for l in lines if int(l.split()[2]) == 2]
        for line in eid2_lines:
            parts = line.split()
            assert parts[4] == "0.000"
            assert parts[5] == "0.000"

    def test_file_still_valid_beltemp_format(self, tmp_path):
        """File written with meshes must still parse cleanly via parse_beltemp."""
        from fahts.core.io.beltemp_parser import parse_beltemp

        mesh = _two_quad_mesh()
        tf = _make_tf_with_section(eid=7, mesh=mesh)
        out = tmp_path / "valid.fem"
        export_beltemp(tf, out, meshes={7: mesh})
        records = parse_beltemp(out)
        assert len(records) == 3  # 3 time steps × 1 element
        for r in records:
            assert r.element_id == 7
