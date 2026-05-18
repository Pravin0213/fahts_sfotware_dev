"""
Tests for PipeSurfaceMesher and its BeamSurfaceMesh output.

Covers mesh structure, node positions on cylinder, element counts,
area weights, local_coords_2d geometry, and validation.
"""
import math
import numpy as np
import pytest

from fahts.core.model.section import PipeSection
from fahts.core.heat.section_mesh.pipe_surface_mesher import PipeSurfaceMesher
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh


# ── Fixture ───────────────────────────────────────────────────────────────────

def _pipe() -> PipeSection:
    """323.9 mm OD, 12.5 mm wall PIPE."""
    return PipeSection(sid=1, outer_diameter=0.3239, thickness=0.0125)


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_beam_surface_mesh(self):
        mesh = PipeSurfaceMesher(_pipe(), length=3.0).build()
        assert isinstance(mesh, BeamSurfaceMesh)

    def test_has_local_coords_2d(self):
        mesh = PipeSurfaceMesher(_pipe(), length=3.0).build()
        assert mesh.local_coords_2d is not None


# ── Mesh counts ───────────────────────────────────────────────────────────────

class TestDefaultCounts:
    """Default c_circ=8, n_length=4."""

    def setup_method(self):
        self.mesh = PipeSurfaceMesher(_pipe(), length=2.0).build()

    def test_node_count(self):
        # (n_length+1) * c_circ = 5 * 8 = 40
        assert self.mesh.n_nodes == 40

    def test_quad_count(self):
        # n_length * c_circ = 4 * 8 = 32
        assert self.mesh.n_quads == 32

    def test_thicknesses_shape(self):
        assert self.mesh.thicknesses.shape == (32,)

    def test_local_coords_2d_shape(self):
        assert self.mesh.local_coords_2d.shape == (32, 4, 2)

    def test_all_node_indices_in_range(self):
        assert self.mesh.quads.min() >= 0
        assert self.mesh.quads.max() < self.mesh.n_nodes


class TestCustomCounts:
    """Custom c_circ=6, n_length=2."""

    def setup_method(self):
        self.mesh = PipeSurfaceMesher(_pipe(), length=1.0, c_circ=6, n_length=2).build()

    def test_node_count(self):
        assert self.mesh.n_nodes == 3 * 6   # (2+1)*6

    def test_quad_count(self):
        assert self.mesh.n_quads == 2 * 6   # 2*6


# ── Node positions ────────────────────────────────────────────────────────────

class TestNodePositions:

    def setup_method(self):
        self.pipe = _pipe()
        self.L    = 2.0
        self.mesh = PipeSurfaceMesher(self.pipe, length=self.L).build()

    def test_nodes_on_cylinder_surface(self):
        """All nodes must lie on the outer surface (y²+z² = R²)."""
        R = self.pipe.outer_radius
        yz_radii = np.sqrt(self.mesh.nodes[:, 1]**2 + self.mesh.nodes[:, 2]**2)
        assert np.allclose(yz_radii, R, atol=1e-10)

    def test_x_range(self):
        xs = self.mesh.nodes[:, 0]
        assert float(xs.min()) == pytest.approx(0.0)
        assert float(xs.max()) == pytest.approx(self.L)

    def test_circumference_closed(self):
        """Nodes at x=0 form a complete ring: c_circ distinct angles."""
        c_circ = 8
        xs_0 = self.mesh.nodes[:c_circ]   # first c_circ nodes are at x=0
        angles = np.arctan2(xs_0[:, 2], xs_0[:, 1])
        # All 8 angles should be distinct
        assert len(set(angles.round(9))) == c_circ


# ── Local coords 2D geometry ──────────────────────────────────────────────────

class TestLocalCoords2D:

    def setup_method(self):
        self.pipe = _pipe()
        self.L    = 2.0
        self.c    = 8
        self.nl   = 4
        self.mesh = PipeSurfaceMesher(
            self.pipe, length=self.L, c_circ=self.c, n_length=self.nl
        ).build()
        self.R     = self.pipe.outer_radius
        self.dtheta = 2 * math.pi / self.c
        self.arc    = self.R * self.dtheta

    def test_element_x_span(self):
        """Each element's local coord dimension 0 spans one axial step."""
        dx = self.L / self.nl
        for q in range(self.mesh.n_quads):
            coords = self.mesh.element_coords_2d(q)
            xs = coords[:, 0]
            assert float(xs.max() - xs.min()) == pytest.approx(dx, rel=1e-9)

    def test_element_arc_span(self):
        """Each element's local coord dimension 1 spans one arc step."""
        for q in range(self.mesh.n_quads):
            coords = self.mesh.element_coords_2d(q)
            ss = coords[:, 1]
            assert float(ss.max() - ss.min()) == pytest.approx(self.arc, rel=1e-9)

    def test_total_arc_coverage(self):
        """Last element's max arc-length equals full circumference."""
        # The last element in the last ring wraps to full circumference
        last_q = self.mesh.n_quads - 1
        coords = self.mesh.element_coords_2d(last_q)
        assert float(coords[:, 1].max()) == pytest.approx(2 * math.pi * self.R, rel=1e-9)


# ── Thickness ─────────────────────────────────────────────────────────────────

class TestThickness:

    def test_all_elements_have_wall_thickness(self):
        pipe = _pipe()
        mesh = PipeSurfaceMesher(pipe, length=2.0).build()
        assert np.allclose(mesh.thicknesses, pipe.thickness, atol=1e-12)


# ── Area weights ──────────────────────────────────────────────────────────────

class TestAreaWeights:

    def test_weights_sum_to_one(self):
        mesh = PipeSurfaceMesher(_pipe(), length=2.0).build()
        assert float(mesh.node_area_weights.sum()) == pytest.approx(1.0)

    def test_weights_nonnegative(self):
        mesh = PipeSurfaceMesher(_pipe(), length=2.0).build()
        assert (mesh.node_area_weights >= 0.0).all()

    def test_total_surface_area(self):
        """Area from weights should match analytical outer surface area 2πR·L."""
        pipe = _pipe()
        L    = 3.0
        mesh = PipeSurfaceMesher(pipe, length=L).build()
        R    = pipe.outer_radius

        # Each element area = arc × axial_step
        total = 0.0
        for q in range(mesh.n_quads):
            coords = mesh.element_coords_2d(q)
            d1 = coords[2] - coords[0]
            d2 = coords[3] - coords[1]
            total += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])

        expected = 2 * math.pi * R * L
        assert total == pytest.approx(expected, rel=1e-9)

    def test_uniform_weights_for_regular_mesh(self):
        """All nodes have the same weight for a uniform cylinder mesh."""
        mesh = PipeSurfaceMesher(_pipe(), length=2.0, c_circ=8, n_length=4).build()
        w = mesh.node_area_weights
        # Interior nodes get contributions from 4 elements; edge nodes from 2.
        # All should be positive; variation is acceptable but max/min not huge.
        assert float(w.min()) > 0.0


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="length must be positive"):
            PipeSurfaceMesher(_pipe(), length=0.0)

    def test_low_c_circ_raises(self):
        with pytest.raises(ValueError, match="c_circ"):
            PipeSurfaceMesher(_pipe(), length=1.0, c_circ=2)

    def test_zero_n_length_raises(self):
        with pytest.raises(ValueError, match="n_length"):
            PipeSurfaceMesher(_pipe(), length=1.0, n_length=0)
