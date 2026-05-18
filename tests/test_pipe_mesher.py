"""Tests for PipeMesher — circular hollow section 2-D FEM mesh generator."""
import math
import numpy as np
import pytest

from fahts.core.heat.section_mesh.pipe_mesher import PipeMesher
from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.model.section import PipeSection


def _make_pipe(sid=1, od=0.3, t=0.02):
    return PipeSection(sid=sid, outer_diameter=od, thickness=t)


class TestPipeMesherBasic:
    def test_returns_section_mesh(self):
        mesh = PipeMesher(_make_pipe()).build()
        assert isinstance(mesh, SectionMesh)

    def test_node_count(self):
        pipe = _make_pipe()
        mesh = PipeMesher(pipe, n_arc=8, n_radial=1).build()
        assert mesh.n_nodes == 2 * 8   # (n_radial+1) * n_arc

    def test_quad_count(self):
        mesh = PipeMesher(_make_pipe(), n_arc=8, n_radial=1).build()
        assert len(mesh.quads) == 8

    def test_outer_edges_count(self):
        mesh = PipeMesher(_make_pipe(), n_arc=8, n_radial=1).build()
        assert mesh.outer_edge_pairs.shape == (8, 2)

    def test_inner_edges_count(self):
        mesh = PipeMesher(_make_pipe(), n_arc=8, n_radial=1).build()
        assert mesh.inner_edge_pairs.shape == (8, 2)

    def test_all_node_indices_valid(self):
        mesh = PipeMesher(_make_pipe()).build()
        n = mesh.n_nodes
        for quad in mesh.quads:
            assert all(0 <= idx < n for idx in quad)


class TestPipeMesherGeometry:
    def test_outer_nodes_at_outer_radius(self):
        pipe = _make_pipe(od=0.3, t=0.02)
        mesh = PipeMesher(pipe, n_arc=8, n_radial=1).build()
        outer_idxs = set(mesh.outer_edge_pairs.flatten())
        R_out = pipe.outer_radius
        for idx in outer_idxs:
            r = math.hypot(mesh.nodes[idx, 0], mesh.nodes[idx, 1])
            assert abs(r - R_out) < 1e-10, f"Outer node {idx} radius={r} != {R_out}"

    def test_inner_nodes_at_inner_radius(self):
        pipe = _make_pipe(od=0.3, t=0.02)
        mesh = PipeMesher(pipe, n_arc=8, n_radial=1).build()
        inner_idxs = set(mesh.inner_edge_pairs.flatten())
        R_in = pipe.inner_radius
        for idx in inner_idxs:
            r = math.hypot(mesh.nodes[idx, 0], mesh.nodes[idx, 1])
            assert abs(r - R_in) < 1e-10, f"Inner node {idx} radius={r} != {R_in}"


class TestPipeMesherAreaConservation:
    def test_total_area_matches_section_area(self):
        pipe = _make_pipe(od=0.3, t=0.02)
        mesh = PipeMesher(pipe, n_arc=32, n_radial=2).build()
        total_area = 0.0
        for quad in mesh.quads:
            coords = mesh.nodes[quad]   # (4, 2)
            # Shoelace formula for a polygon (exact for straight-sided quads)
            x, y = coords[:, 0], coords[:, 1]
            area = 0.5 * abs(sum(
                x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i]
                for i in range(4)
            ))
            total_area += area
        expected = pipe.cross_section_area
        # Polygon approximation of annulus: ~0.6% error for n_arc=32
        assert abs(total_area - expected) / expected < 0.01


class TestPipeMesherValidation:
    def test_low_n_arc_raises(self):
        with pytest.raises(ValueError, match="n_arc"):
            PipeMesher(_make_pipe(), n_arc=2)

    def test_low_n_radial_raises(self):
        with pytest.raises(ValueError, match="n_radial"):
            PipeMesher(_make_pipe(), n_radial=0)

    def test_zero_inner_radius_raises(self):
        pipe = PipeSection(sid=1, outer_diameter=0.3, thickness=0.15)  # solid
        with pytest.raises(ValueError, match="inner radius"):
            PipeMesher(pipe)


class TestPipeMesherMultiRadial:
    def test_n_radial_2_node_count(self):
        mesh = PipeMesher(_make_pipe(), n_arc=8, n_radial=2).build()
        assert mesh.n_nodes == 3 * 8  # (2+1)*8

    def test_n_radial_2_quad_count(self):
        mesh = PipeMesher(_make_pipe(), n_arc=8, n_radial=2).build()
        assert len(mesh.quads) == 2 * 8  # 2 layers × 8 arcs
