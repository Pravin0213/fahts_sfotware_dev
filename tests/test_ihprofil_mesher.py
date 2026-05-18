"""Tests for IProfileMesher — I/H-section 2-D FEM mesh generator."""
import numpy as np
import pytest

from fahts.core.heat.section_mesh.ihprofil_mesher import IProfileMesher
from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.model.section import ISection


def _make_i(sid=1, h=0.5, tw=0.02, bf_top=0.2, tf_top=0.02, bf_bot=0.2, tf_bot=0.02):
    return ISection(
        sid=sid, h=h, tw=tw,
        bf_top=bf_top, tf_top=tf_top,
        bf_bot=bf_bot, tf_bot=tf_bot,
    )


class TestIProfileMesherBasic:
    def test_returns_section_mesh(self):
        mesh = IProfileMesher(_make_i()).build()
        assert isinstance(mesh, SectionMesh)

    def test_has_nodes_and_quads(self):
        mesh = IProfileMesher(_make_i()).build()
        assert mesh.nodes.shape[1] == 2
        assert mesh.quads.shape[1] == 4
        assert len(mesh.quads) > 0

    def test_outer_edges_not_empty(self):
        mesh = IProfileMesher(_make_i()).build()
        assert mesh.outer_edge_pairs.shape[0] > 0

    def test_inner_edges_empty(self):
        mesh = IProfileMesher(_make_i()).build()
        assert mesh.inner_edge_pairs.shape[0] == 0

    def test_all_node_indices_valid(self):
        mesh = IProfileMesher(_make_i()).build()
        n = len(mesh.nodes)
        for quad in mesh.quads:
            assert all(0 <= idx < n for idx in quad)
        for a, b in mesh.outer_edge_pairs:
            assert 0 <= a < n and 0 <= b < n

    def test_no_duplicate_quads(self):
        mesh = IProfileMesher(_make_i()).build()
        seen = set()
        for q in mesh.quads:
            key = tuple(sorted(q))
            assert key not in seen, f"Duplicate quad: {key}"
            seen.add(key)

    def test_nodes_within_section_bounds(self):
        s = _make_i(h=0.5, tw=0.02, bf_top=0.2, tf_top=0.02, bf_bot=0.2, tf_bot=0.02)
        mesh = IProfileMesher(s).build()
        y, z = mesh.nodes[:, 0], mesh.nodes[:, 1]
        assert float(y.min()) >= -s.bf_top / 2 - 1e-9
        assert float(y.max()) <= s.bf_top / 2 + 1e-9
        assert float(z.min()) >= -s.h / 2 - 1e-9
        assert float(z.max()) <= s.h / 2 + 1e-9


class TestIProfileMesherAreaConservation:
    def test_total_quad_area_matches_section_area(self):
        s = _make_i(h=0.5, tw=0.02, bf_top=0.2, tf_top=0.02, bf_bot=0.2, tf_bot=0.02)
        mesh = IProfileMesher(s).build()
        total_area = 0.0
        for quad in mesh.quads:
            coords = mesh.nodes[quad]
            v1 = coords[1] - coords[0]
            v2 = coords[3] - coords[0]
            v3 = coords[2] - coords[0]
            a1 = abs(v1[0] * v2[1] - v1[1] * v2[0]) * 0.5
            a2 = abs(v2[0] * v3[1] - v2[1] * v3[0]) * 0.5
            total_area += a1 + a2
        assert abs(total_area - s.cross_section_area) < 1e-6


class TestIProfileMesherNLayers:
    def test_n_layers_1(self):
        m1 = IProfileMesher(_make_i(), n_layers=1).build()
        assert len(m1.quads) > 0

    def test_n_layers_2_has_more_quads(self):
        m1 = IProfileMesher(_make_i(), n_layers=1).build()
        m2 = IProfileMesher(_make_i(), n_layers=2).build()
        assert len(m2.quads) >= len(m1.quads)

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            IProfileMesher(_make_i(), n_layers=0)


class TestIProfileMesherValidation:
    def test_zero_web_height_raises(self):
        s = ISection(sid=1, h=0.04, tw=0.01, bf_top=0.1, tf_top=0.02,
                     bf_bot=0.1, tf_bot=0.02)
        # h=0.04, tf_top=0.02, tf_bot=0.02 → web_height=0
        with pytest.raises(ValueError, match="web height"):
            IProfileMesher(s)

    def test_asymmetric_flanges(self):
        s = ISection(sid=2, h=0.6, tw=0.015, bf_top=0.25, tf_top=0.025,
                     bf_bot=0.15, tf_bot=0.015)
        mesh = IProfileMesher(s).build()
        assert len(mesh.quads) > 0
