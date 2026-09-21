"""
BELTEMP export: §3.4.2 equivalent thermal-expansion linearization tests.

Verifies that export_beltemp() writes correct incremental βy/βz values
derived from the §3.4.2 first-moment formula for both SectionMesh (legacy
2-D cross-section) and BeamSurfaceMesh (active 3-D surface mesh).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.io.beltemp_parser import cumulative_temperatures, parse_beltemp
from fahts.core.io.results_writer import export_beltemp
from fahts.core.results.temperature_field import TemperatureField


# ── Mesh helpers ─────────────────────────────────────────────────────────────

def _section_mesh_2d(half_y: float = 0.1, half_z: float = 0.2) -> SectionMesh:
    """2×2 quad SectionMesh in [y, z] centred at origin."""
    nodes = np.array(
        [
            [-half_y, -half_z],
            [ half_y, -half_z],
            [ half_y,  half_z],
            [-half_y,  half_z],
        ],
        dtype=float,
    )
    quads = np.array([[0, 1, 2, 3]], dtype=int)
    return SectionMesh(
        nodes=nodes,
        quads=quads,
        outer_edge_pairs=np.empty((0, 2), dtype=int),
        inner_edge_pairs=np.empty((0, 2), dtype=int),
    )


def _beam_surface_mesh_3d(
    beam_length: float = 2.0,
    half_y: float = 0.1,
    half_z: float = 0.2,
    thickness: float = 0.008,
) -> BeamSurfaceMesh:
    """
    Minimal 1-slab (2 quads per face = 8 quads) 3-D surface mesh for a BOX beam.

    Nodes: [x, y, z] with x ∈ {0, beam_length}, y ∈ {±half_y}, z ∈ {±half_z}.
    """
    pool: list[np.ndarray] = []
    idx: dict[tuple, int] = {}

    def gid(x: float, y: float, z: float) -> int:
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in idx:
            idx[key] = len(pool)
            pool.append(np.array([x, y, z], dtype=float))
        return idx[key]

    hy, hz, L = half_y, half_z, beam_length
    quads_list = [
        # Bottom (z = -hz): two quads split at x = L/2
        (gid(0,   -hy, -hz), gid(L/2, -hy, -hz), gid(L/2,  hy, -hz), gid(0,    hy, -hz)),
        (gid(L/2, -hy, -hz), gid(L,   -hy, -hz), gid(L,    hy, -hz), gid(L/2,  hy, -hz)),
        # Top (z = +hz)
        (gid(0,    hy,  hz), gid(L/2,  hy,  hz), gid(L/2, -hy,  hz), gid(0,   -hy,  hz)),
        (gid(L/2,  hy,  hz), gid(L,    hy,  hz), gid(L,   -hy,  hz), gid(L/2, -hy,  hz)),
        # Left (y = -hy)
        (gid(0,   -hy, -hz), gid(0,   -hy,  hz), gid(L/2, -hy,  hz), gid(L/2, -hy, -hz)),
        (gid(L/2, -hy, -hz), gid(L/2, -hy,  hz), gid(L,   -hy,  hz), gid(L,   -hy, -hz)),
        # Right (y = +hy)
        (gid(0,    hy,  hz), gid(0,    hy, -hz), gid(L/2,  hy, -hz), gid(L/2,  hy,  hz)),
        (gid(L/2,  hy,  hz), gid(L/2,  hy, -hz), gid(L,    hy, -hz), gid(L,    hy,  hz)),
    ]

    n_q = len(quads_list)
    return BeamSurfaceMesh(
        nodes=np.array(pool, dtype=float),
        quads=np.array(quads_list, dtype=np.intp),
        thicknesses=np.full(n_q, thickness, dtype=float),
        local_cols=np.zeros((n_q, 2), dtype=np.intp),
    )


def _write_and_parse(
    field: TemperatureField,
    meshes: dict | None,
    T_initial: float = 20.0,
) -> dict:
    """Export BELTEMP to a temp file and return cumulative_temperatures dict."""
    with tempfile.NamedTemporaryFile(suffix=".fem", delete=False, mode="w") as fh:
        tmp = Path(fh.name)
    export_beltemp(field, tmp, T_initial=T_initial, meshes=meshes)
    records = parse_beltemp(tmp)
    tmp.unlink(missing_ok=True)
    return cumulative_temperatures(records, T_initial=T_initial)


# ── Tests: SectionMesh path ───────────────────────────────────────────────────

class TestBeltempLinearizationSectionMesh:
    """BELTEMP §3.4.2 export with legacy 2-D SectionMesh."""

    def test_zero_gradient_when_no_mesh_supplied(self):
        """Without a mesh dict, βy = βz = 0 at every step."""
        field = TemperatureField(
            times=np.array([60.0, 120.0]),
            element_ids=[1],
            T_centroid=np.array([[50.0], [80.0]]),
            T_section={},
        )
        cum = _write_and_parse(field, meshes=None)
        _, _, T_gy, T_gz = cum[1]
        np.testing.assert_allclose(T_gy, 0.0, atol=1e-10)
        np.testing.assert_allclose(T_gz, 0.0, atol=1e-10)

    def test_mean_temperature_increments_match_centroid(self):
        """Exported ΔT_mean matches T_centroid increments."""
        T_initial = 20.0
        T_steps = np.array([[45.0], [70.0], [95.0]])
        field = TemperatureField(
            times=np.array([60.0, 120.0, 180.0]),
            element_ids=[5],
            T_centroid=T_steps,
            T_section={},
        )
        cum = _write_and_parse(field, meshes=None, T_initial=T_initial)
        _, T_mean, _, _ = cum[5]
        np.testing.assert_allclose(T_mean[0], T_initial, atol=1e-9)
        np.testing.assert_allclose(T_mean[1:], T_steps[:, 0], rtol=1e-6)

    def test_section_mesh_gradient_roundtrip(self):
        """
        Linear T(y, z) on SectionMesh → BELTEMP → parse: βy/βz recover from
        the §3.4.2 first-moment formula.
        """
        sm = _section_mesh_2d(half_y=0.1, half_z=0.2)
        beta_y_true = -8.0
        beta_z_true = 20.0
        T_initial = 20.0

        # Two time steps: step 1 at half gradient, step 2 at full gradient
        def T_at(scale: float) -> np.ndarray:
            return (T_initial
                    + scale * beta_y_true * sm.nodes[:, 1]
                    + scale * beta_z_true * sm.nodes[:, 0])

        T1 = T_at(0.5)
        T2 = T_at(1.0)
        T_c1 = float(T1.mean())
        T_c2 = float(T2.mean())

        field = TemperatureField(
            times=np.array([60.0, 120.0]),
            element_ids=[7],
            T_centroid=np.array([[T_c1], [T_c2]]),
            T_section={7: np.array([T1, T2])},
        )
        cum = _write_and_parse(field, meshes={7: sm}, T_initial=T_initial)
        _, _, T_gy, T_gz = cum[7]

        # Cumulative values at last step equal the full gradient
        assert T_gy[-1] == pytest.approx(beta_y_true, rel=1e-6)
        assert T_gz[-1] == pytest.approx(beta_z_true, rel=1e-6)


# ── Tests: BeamSurfaceMesh path ───────────────────────────────────────────────

class TestBeltempLinearizationBeamSurfaceMesh:
    """BELTEMP §3.4.2 export with the active 3-D BeamSurfaceMesh."""

    def test_zero_gradient_for_uniform_temperature(self):
        """Uniform T → βy = βz = 0 even with a BeamSurfaceMesh supplied."""
        bsm = _beam_surface_mesh_3d()
        T_uniform = np.full(bsm.n_nodes, 120.0)
        field = TemperatureField(
            times=np.array([60.0]),
            element_ids=[1],
            T_centroid=np.array([[120.0]]),
            T_section={1: T_uniform[None, :]},
        )
        cum = _write_and_parse(field, meshes={1: bsm})
        _, _, T_gy, T_gz = cum[1]
        assert T_gy[-1] == pytest.approx(0.0, abs=1e-9)
        assert T_gz[-1] == pytest.approx(0.0, abs=1e-9)

    def test_linear_T_y_recovers_beta_z(self):
        """T = T0 + βz·y on 3-D surface → BELTEMP βz matches exactly."""
        bsm = _beam_surface_mesh_3d(beam_length=2.0, half_y=0.1, half_z=0.2)
        beta_z_true = 30.0
        T_initial = 20.0

        T_nodes = T_initial + beta_z_true * bsm.nodes[:, 1]
        T_centroid = float(np.average(
            T_nodes,
            weights=bsm.node_area_weights,
        ))

        field = TemperatureField(
            times=np.array([60.0]),
            element_ids=[2],
            T_centroid=np.array([[T_centroid]]),
            T_section={2: T_nodes[None, :]},
        )
        cum = _write_and_parse(field, meshes={2: bsm}, T_initial=T_initial)
        _, _, T_gy, T_gz = cum[2]

        assert T_gz[-1] == pytest.approx(beta_z_true, rel=1e-6)
        assert T_gy[-1] == pytest.approx(0.0, abs=1e-6)

    def test_linear_T_z_recovers_beta_y(self):
        """T = T0 + βy·z on 3-D surface → BELTEMP βy matches exactly."""
        bsm = _beam_surface_mesh_3d(beam_length=2.0, half_y=0.1, half_z=0.2)
        beta_y_true = -15.0
        T_initial = 20.0

        T_nodes = T_initial + beta_y_true * bsm.nodes[:, 2]
        T_centroid = float(np.average(T_nodes, weights=bsm.node_area_weights))

        field = TemperatureField(
            times=np.array([60.0]),
            element_ids=[3],
            T_centroid=np.array([[T_centroid]]),
            T_section={3: T_nodes[None, :]},
        )
        cum = _write_and_parse(field, meshes={3: bsm}, T_initial=T_initial)
        _, _, T_gy, T_gz = cum[3]

        assert T_gy[-1] == pytest.approx(beta_y_true, rel=1e-6)
        assert T_gz[-1] == pytest.approx(0.0, abs=1e-6)

    def test_combined_gradient_both_axes(self):
        """T = T0 + βy·z + βz·y → both βy and βz recovered correctly."""
        bsm = _beam_surface_mesh_3d(beam_length=3.0, half_y=0.15, half_z=0.25)
        beta_y_true = -10.0
        beta_z_true = 18.0
        T_initial = 20.0

        T_nodes = (T_initial
                   + beta_y_true * bsm.nodes[:, 2]
                   + beta_z_true * bsm.nodes[:, 1])
        T_centroid = float(np.average(T_nodes, weights=bsm.node_area_weights))

        field = TemperatureField(
            times=np.array([60.0]),
            element_ids=[4],
            T_centroid=np.array([[T_centroid]]),
            T_section={4: T_nodes[None, :]},
        )
        cum = _write_and_parse(field, meshes={4: bsm}, T_initial=T_initial)
        _, _, T_gy, T_gz = cum[4]

        assert T_gy[-1] == pytest.approx(beta_y_true, rel=1e-6)
        assert T_gz[-1] == pytest.approx(beta_z_true, rel=1e-6)

    def test_incremental_gradient_matches_step_differences(self):
        """
        Over two time steps, the cumulative βy at step 2 equals the sum of
        the incremental values written in the BELTEMP file.
        """
        bsm = _beam_surface_mesh_3d()
        T_initial = 20.0
        beta_z_step1 = 5.0
        beta_z_step2 = 12.0

        T1 = T_initial + beta_z_step1 * bsm.nodes[:, 1]
        T2 = T_initial + beta_z_step2 * bsm.nodes[:, 1]
        T_c1 = float(np.average(T1, weights=bsm.node_area_weights))
        T_c2 = float(np.average(T2, weights=bsm.node_area_weights))

        field = TemperatureField(
            times=np.array([60.0, 120.0]),
            element_ids=[5],
            T_centroid=np.array([[T_c1], [T_c2]]),
            T_section={5: np.array([T1, T2])},
        )
        cum = _write_and_parse(field, meshes={5: bsm}, T_initial=T_initial)
        _, _, T_gy, T_gz = cum[5]

        assert T_gz[1] == pytest.approx(beta_z_step1, rel=1e-6)
        assert T_gz[2] == pytest.approx(beta_z_step2, rel=1e-6)

    def test_element_without_mesh_gets_zero_gradient(self):
        """If a mesh entry is missing, that element's βy/βz stay zero."""
        bsm = _beam_surface_mesh_3d()
        T_initial = 20.0
        beta_z = 25.0

        T_nodes = T_initial + beta_z * bsm.nodes[:, 1]
        T_centroid = float(np.average(T_nodes, weights=bsm.node_area_weights))

        field = TemperatureField(
            times=np.array([60.0]),
            element_ids=[10, 11],
            T_centroid=np.array([[T_centroid, T_centroid]]),
            T_section={
                10: T_nodes[None, :],
                11: T_nodes[None, :],
            },
        )
        # Only eid=10 has a mesh entry; eid=11 should get zero gradients
        cum = _write_and_parse(field, meshes={10: bsm}, T_initial=T_initial)
        _, _, T_gy_10, T_gz_10 = cum[10]
        _, _, T_gy_11, T_gz_11 = cum[11]

        assert T_gz_10[-1] == pytest.approx(beta_z, rel=1e-6)
        assert T_gz_11[-1] == pytest.approx(0.0, abs=1e-9)
        assert T_gy_11[-1] == pytest.approx(0.0, abs=1e-9)

    def test_section_gradient_uses_3d_face_area_not_projected(self):
        """
        On a non-axis-aligned face the 3-D area (cross product) is used,
        not the projection.  Verify by checking the moment-of-inertia
        denominator Iz is positive for a symmetric box.
        """
        bsm = _beam_surface_mesh_3d(half_y=0.05, half_z=0.1)
        T_nodes = 20.0 + 10.0 * bsm.nodes[:, 1]
        field = TemperatureField(
            times=np.array([1.0]),
            element_ids=[99],
            T_centroid=np.array([[20.0]]),
            T_section={99: T_nodes[None, :]},
        )
        by, bz = field.section_gradient(99, 0, bsm)
        assert bz == pytest.approx(10.0, rel=1e-6)
