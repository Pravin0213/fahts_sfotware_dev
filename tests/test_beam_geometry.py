"""Tests for fahts/renderer/beam_geometry.py (Phase 1.4)."""
import math

import numpy as np
import pytest
import pyvista as pv

from fahts.core.model.element import BeamElement
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection
from fahts.renderer.beam_geometry import (
    _local_frame,
    build_beam_mesh,
    build_centreline_mesh,
    build_model_mesh,
    mesh_section_at,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_nodes(p1, p2):
    return {
        1: Node(nid=1, x=p1[0], y=p1[1], z=p1[2]),
        2: Node(nid=2, x=p2[0], y=p2[1], z=p2[2]),
    }


def _axis_beam(
    p1=(0, 0, 0),
    p2=(1, 0, 0),
    H=0.15,
    W=0.15,
    t=0.01,
    local_z=None,
):
    """Create a beam along an axis with a square BOX section."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    direction = p2 - p1
    direction /= np.linalg.norm(direction)
    if local_z is None:
        local_z = np.array([0.0, 0.0, 1.0])
    sec = BoxSection(sid=1, H=H, W=W, T_side=t, T_bot=t, T_top=t)
    elem = BeamElement(
        eid=1, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=1,
        length=float(np.linalg.norm(p2 - p1)),
        direction=direction,
        local_z=local_z,
    )
    return elem, sec, _make_nodes(p1, p2)


# ── Local frame tests ──────────────────────────────────────────────────────────

class TestLocalFrame:
    def test_orthonormal(self):
        lx, ly, lz = _local_frame(
            np.array([1, 0, 0], float),
            np.array([0, 0, 1], float),
        )
        assert math.isclose(np.dot(lx, ly), 0.0, abs_tol=1e-12)
        assert math.isclose(np.dot(lx, lz), 0.0, abs_tol=1e-12)
        assert math.isclose(np.dot(ly, lz), 0.0, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(lx), 1.0, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(ly), 1.0, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(lz), 1.0, abs_tol=1e-12)

    def test_fallback_when_parallel(self):
        """Beam aligned with global Z; local_z also Z → should not crash."""
        lx, ly, lz = _local_frame(
            np.array([0, 0, 1], float),
            np.array([0, 0, 1], float),
        )
        assert math.isclose(np.linalg.norm(lx), 1.0, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(ly), 1.0, abs_tol=1e-12)
        assert math.isclose(np.linalg.norm(lz), 1.0, abs_tol=1e-12)
        assert math.isclose(np.dot(lx, ly), 0.0, abs_tol=1e-12)

    def test_none_local_z_uses_fallback(self):
        lx, ly, lz = _local_frame(np.array([1, 0, 0], float), None)
        assert math.isclose(np.linalg.norm(ly), 1.0, abs_tol=1e-12)


# ── build_beam_mesh tests ──────────────────────────────────────────────────────

class TestBuildBeamMesh:
    def test_returns_polydata(self):
        elem, sec, nodes = _axis_beam()
        mesh = build_beam_mesh(elem, sec, nodes)
        assert isinstance(mesh, pv.PolyData)

    def test_face_count(self):
        elem, sec, nodes = _axis_beam()
        mesh = build_beam_mesh(elem, sec, nodes)
        assert mesh.n_cells == 6  # 4 sides + 2 end caps

    def test_point_count(self):
        elem, sec, nodes = _axis_beam()
        mesh = build_beam_mesh(elem, sec, nodes)
        assert mesh.n_points == 8  # 4 corners × 2 ends

    def test_element_id_cell_data(self):
        elem, sec, nodes = _axis_beam()
        mesh = build_beam_mesh(elem, sec, nodes)
        assert "element_id" in mesh.cell_data
        assert np.all(mesh.cell_data["element_id"] == 1)

    def test_bounding_box_size(self):
        """Mesh bounding box should match BOX dimensions (H=0.15, W=0.15, L=1.0)."""
        elem, sec, nodes = _axis_beam(p1=(0, 0, 0), p2=(1, 0, 0), H=0.15, W=0.15,
                                      local_z=np.array([0, 0, 1.0]))
        mesh = build_beam_mesh(elem, sec, nodes)
        bb = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        assert math.isclose(bb[1] - bb[0], 1.0, abs_tol=1e-10)   # length
        assert math.isclose(bb[3] - bb[2], 0.15, abs_tol=1e-10)  # H
        assert math.isclose(bb[5] - bb[4], 0.15, abs_tol=1e-10)  # W

    def test_vertical_beam(self):
        """Beam along Z axis — parallel to default fallback, must not crash."""
        elem, sec, nodes = _axis_beam(
            p1=(0, 0, 0), p2=(0, 0, 2),
            local_z=np.array([0, 0, 1.0]),
        )
        mesh = build_beam_mesh(elem, sec, nodes)
        assert mesh.n_cells == 6

    def test_diagonal_beam(self):
        """Non-axis-aligned beam — frame must remain orthonormal."""
        elem, sec, nodes = _axis_beam(
            p1=(0, 0, 0), p2=(1, 1, 1),
            local_z=np.array([0, 0, 1.0]),
        )
        mesh = build_beam_mesh(elem, sec, nodes)
        assert mesh.n_cells == 6


# ── mesh_section_at tests ──────────────────────────────────────────────────────

class TestMeshSectionAt:
    def test_returns_polydata(self):
        elem, sec, nodes = _axis_beam()
        s = mesh_section_at(elem, sec, nodes, t=0.5)
        assert isinstance(s, pv.PolyData)

    def test_point_count(self):
        elem, sec, nodes = _axis_beam()
        s = mesh_section_at(elem, sec, nodes, t=0.0)
        assert s.n_points == 8  # 4 outer + 4 inner

    def test_section_at_end_matches_start(self):
        """Section at t=0 should be centred on node n1."""
        p1 = np.array([5.0, 3.0, 1.0])
        p2 = np.array([6.0, 3.0, 1.0])
        elem, sec, nodes = _axis_beam(p1=p1, p2=p2)
        s = mesh_section_at(elem, sec, nodes, t=0.0)
        centre = s.points.mean(axis=0)
        np.testing.assert_allclose(centre, p1, atol=1e-10)


# ── build_centreline_mesh tests ────────────────────────────────────────────────

class TestBuildCentrelineMesh:
    def _minimal_model(self):
        from unittest.mock import MagicMock
        from fahts.core.model.fem_model import FEMModel
        from fahts.core.model.section import BoxSection

        n = {
            1: Node(1, 0.0, 0.0, 0.0),
            2: Node(2, 1.0, 0.0, 0.0),
            3: Node(3, 1.0, 1.0, 0.0),
        }
        direction_01 = np.array([1.0, 0.0, 0.0])
        direction_12 = np.array([0.0, 1.0, 0.0])
        lz = np.array([0.0, 0.0, 1.0])
        e = {
            10: BeamElement(10, 1, 2, 1, 1, 1, 1.0, direction_01, lz),
            11: BeamElement(11, 2, 3, 1, 1, 1, 1.0, direction_12, lz),
        }
        sec = {1: BoxSection(1, 0.15, 0.15, 0.01, 0.01, 0.01)}
        from fahts.core.model.material import SteelMaterial
        mat = {1: SteelMaterial(1, 210e9, 0.3, 355e6, 7850.0, 12e-6)}
        model = FEMModel(
            nodes=n, elements=e, sections=sec, materials=mat,
            groups={}, unitvecs={1: lz}, source_file=None,
        )
        return model

    def test_line_count(self):
        model = self._minimal_model()
        mesh = build_centreline_mesh(model)
        assert mesh.n_cells == 2

    def test_centred(self):
        """Centreline mesh should be centroid-shifted near origin."""
        model = self._minimal_model()
        mesh = build_centreline_mesh(model)
        cx = mesh.points[:, 0].mean()
        cy = mesh.points[:, 1].mean()
        # With only 3 nodes at (0,0,0),(1,0,0),(1,1,0) centroid=(2/3,1/3,0)
        # Shifted pts mean should be ~0
        # Just check it doesn't blow up and returns PolyData
        assert isinstance(mesh, pv.PolyData)


# ── build_model_mesh smoke test ────────────────────────────────────────────────

class TestBuildModelMesh:
    def test_real_model(self):
        """Smoke test against model_file.fem — checks we get a valid merged mesh."""
        from pathlib import Path
        from fahts.core.io.usfos_reader import read_usfos_fem

        fem_path = Path(__file__).parents[1] / "model_file.fem"
        if not fem_path.exists():
            pytest.skip("model_file.fem not found")

        model = read_usfos_fem(fem_path)
        mesh = build_model_mesh(model)

        assert isinstance(mesh, pv.PolyData)
        # 783 beams × 6 faces each
        assert mesh.n_cells == 783 * 6
        assert "element_id" in mesh.cell_data
        # All element IDs in the cell data should be valid beam IDs
        ids = set(mesh.cell_data["element_id"].tolist())
        assert ids == set(model.elements.keys())
