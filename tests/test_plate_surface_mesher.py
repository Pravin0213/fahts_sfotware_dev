"""
Tests for PlateSurfaceMesher and its BeamSurfaceMesh output.

Covers mesh structure, node interpolation, local_coords_2d, area, and validation.
"""
import math
import numpy as np
import pytest

from fahts.core.model.section import PlateSection
from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _plate() -> PlateSection:
    return PlateSection(sid=1, thickness=0.010)


def _square_corners() -> np.ndarray:
    """1 m × 1 m flat plate in the XY-plane."""
    return np.array([
        [0.0, 0.0, 0.0],  # P1
        [1.0, 0.0, 0.0],  # P2
        [1.0, 1.0, 0.0],  # P3
        [0.0, 1.0, 0.0],  # P4
    ])


def _tilted_corners() -> np.ndarray:
    """2 m × 1 m plate tilted 45° in XZ-plane."""
    c, s = math.cos(math.pi/4), math.sin(math.pi/4)
    return np.array([
        [0.0, 0.0, 0.0],
        [2*c, 0.0, 2*s],   # P2 = P1 + 2m along (cos45, 0, sin45)
        [2*c, 1.0, 2*s],   # P3 = P2 + 1m in y
        [0.0, 1.0, 0.0],   # P4 = P1 + 1m in y
    ])


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_beam_surface_mesh(self):
        mesh = PlateSurfaceMesher(_plate(), _square_corners()).build()
        assert isinstance(mesh, BeamSurfaceMesh)

    def test_has_local_coords_2d(self):
        mesh = PlateSurfaceMesher(_plate(), _square_corners()).build()
        assert mesh.local_coords_2d is not None


# ── Mesh counts ───────────────────────────────────────────────────────────────

class TestDefaultCounts:
    """Default mesh_12=4, mesh_14=2."""

    def setup_method(self):
        self.mesh = PlateSurfaceMesher(_plate(), _square_corners()).build()

    def test_node_count(self):
        # (mesh_14+1) × (mesh_12+1) = 3 × 5 = 15
        assert self.mesh.n_nodes == 15

    def test_quad_count(self):
        # mesh_14 × mesh_12 = 2 × 4 = 8
        assert self.mesh.n_quads == 8

    def test_thicknesses_shape(self):
        assert self.mesh.thicknesses.shape == (8,)

    def test_local_coords_2d_shape(self):
        assert self.mesh.local_coords_2d.shape == (8, 4, 2)

    def test_all_node_indices_in_range(self):
        assert self.mesh.quads.min() >= 0
        assert self.mesh.quads.max() < self.mesh.n_nodes


class TestCustomCounts:
    """Custom mesh_12=3, mesh_14=3."""

    def setup_method(self):
        self.mesh = PlateSurfaceMesher(
            _plate(), _square_corners(), mesh_12=3, mesh_14=3
        ).build()

    def test_node_count(self):
        assert self.mesh.n_nodes == 4 * 4   # (3+1)²

    def test_quad_count(self):
        assert self.mesh.n_quads == 3 * 3   # 3²


# ── Node positions ────────────────────────────────────────────────────────────

class TestNodePositions:

    def test_corner_nodes_match_input_for_square_plate(self):
        """Corner nodes of the mesh must coincide with the 4 input corners."""
        corners = _square_corners()
        mesh = PlateSurfaceMesher(_plate(), corners, mesh_12=2, mesh_14=2).build()

        def has_node_near(P):
            return any(np.linalg.norm(mesh.nodes[i] - P) < 1e-9
                       for i in range(mesh.n_nodes))

        for P in corners:
            assert has_node_near(P), f"Corner {P} not found in mesh"

    def test_nodes_in_plate_plane(self):
        """For a flat plate all nodes must lie in the same plane."""
        corners = _square_corners()
        mesh = PlateSurfaceMesher(_plate(), corners).build()
        # All z-coordinates should be 0 for XY-plane plate
        assert np.allclose(mesh.nodes[:, 2], 0.0, atol=1e-10)

    def test_tilted_plate_nodes_in_plane(self):
        """Nodes for a tilted plate should lie on the tilted plane."""
        corners = _tilted_corners()
        mesh = PlateSurfaceMesher(_plate(), corners).build()
        # Normal to tilted plate: cross(P2-P1, P4-P1)
        P1, P2, _, P4 = corners
        normal = np.cross(P2 - P1, P4 - P1)
        normal /= np.linalg.norm(normal)
        # All nodes should satisfy dot(node - P1, normal) ≈ 0
        for node in mesh.nodes:
            assert abs(float(np.dot(node - P1, normal))) < 1e-9


# ── Local 2D coordinates ──────────────────────────────────────────────────────

class TestLocalCoords2D:

    def test_corner_element_local_coords_rectangular_plate(self):
        """
        For a 1×1 plate with mesh_12=4, mesh_14=2:
        Bottom-left element (q=0) should span (0,0) to (0.25, 0.5) in local coords.
        """
        mesh = PlateSurfaceMesher(
            _plate(), _square_corners(), mesh_12=4, mesh_14=2
        ).build()
        coords = mesh.element_coords_2d(0)
        assert float(coords[:, 0].min()) == pytest.approx(0.0, abs=1e-9)
        assert float(coords[:, 1].min()) == pytest.approx(0.0, abs=1e-9)

    def test_total_extent_matches_plate_dimensions(self):
        """Local coord range should match plate dimensions."""
        L12 = 2.0  # length along edge 1-2
        L14 = 1.0  # length along edge 1-4
        corners = np.array([
            [0.0, 0.0, 0.0],
            [L12, 0.0, 0.0],
            [L12, L14, 0.0],
            [0.0, L14, 0.0],
        ])
        mesh = PlateSurfaceMesher(_plate(), corners).build()
        all_e1 = mesh.local_coords_2d[:, :, 0].ravel()
        all_e2 = mesh.local_coords_2d[:, :, 1].ravel()
        assert float(all_e1.max()) == pytest.approx(L12, rel=1e-9)
        assert float(all_e2.max()) == pytest.approx(L14, rel=1e-9)


# ── Thickness ─────────────────────────────────────────────────────────────────

class TestThickness:
    def test_all_elements_have_plate_thickness(self):
        plate = _plate()
        mesh  = PlateSurfaceMesher(plate, _square_corners()).build()
        assert np.allclose(mesh.thicknesses, plate.thickness, atol=1e-12)


# ── Area weights ──────────────────────────────────────────────────────────────

class TestAreaWeights:

    def test_weights_sum_to_one(self):
        mesh = PlateSurfaceMesher(_plate(), _square_corners()).build()
        assert float(mesh.node_area_weights.sum()) == pytest.approx(1.0)

    def test_weights_nonnegative(self):
        mesh = PlateSurfaceMesher(_plate(), _square_corners()).build()
        assert (mesh.node_area_weights >= 0.0).all()

    def test_total_area_rectangular_plate(self):
        """Sum of element areas should equal the plate face area."""
        L12 = 3.0
        L14 = 2.0
        corners = np.array([
            [0.0, 0.0, 0.0],
            [L12, 0.0, 0.0],
            [L12, L14, 0.0],
            [0.0, L14, 0.0],
        ])
        mesh = PlateSurfaceMesher(_plate(), corners, mesh_12=3, mesh_14=2).build()

        total = 0.0
        for q in range(mesh.n_quads):
            coords = mesh.element_coords_2d(q)
            d1 = coords[2] - coords[0]
            d2 = coords[3] - coords[1]
            total += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])

        assert total == pytest.approx(L12 * L14, rel=1e-9)


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:

    def test_wrong_corners_shape_raises(self):
        with pytest.raises(ValueError, match="corners must be shape"):
            PlateSurfaceMesher(_plate(), np.zeros((3, 3)))

    def test_zero_mesh_12_raises(self):
        with pytest.raises(ValueError, match="mesh_12"):
            PlateSurfaceMesher(_plate(), _square_corners(), mesh_12=0)

    def test_zero_mesh_14_raises(self):
        with pytest.raises(ValueError, match="mesh_14"):
            PlateSurfaceMesher(_plate(), _square_corners(), mesh_14=0)

    def test_degenerate_plate_raises(self):
        """Collinear nodes should raise ValueError."""
        collinear = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear
            [0.0, 0.0, 0.0],
        ])
        with pytest.raises(ValueError, match="Degenerate"):
            PlateSurfaceMesher(_plate(), collinear).build()
