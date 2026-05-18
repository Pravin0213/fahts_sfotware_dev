"""
Unit tests for Task 3.2 — BOX cross-section 2-D mesh generator.

For a symmetric BOX(H=0.20, W=0.20, T=0.008) with elem_size=0.04 and n_layers=1:

  y-grid (plates): y_L=[−0.1, −0.092]  y_M=[−0.092..0.092, 6 pts]  y_R=[0.092, 0.1]
                   → y_plate: 8 points, 7 segments per plate row

  z-grid:  z_bot=[−0.1, −0.092]   z_web=[−0.092..0.092, 6 pts]   z_top=[0.092, 0.1]

  Quads: bot_plate=7, top_plate=7, left_web=5, right_web=5  → 24 total
  Nodes: 16(bot) + 16(top) + 8(left new) + 8(right new) = 48

  Outer perimeter: 2*(0.2+0.2) = 0.8 m  (28 edges)
  Inner perimeter: 2*(0.184+0.184) = 0.736 m  (20 edges)
  Steel area:      0.2^2 − 0.184^2 = 0.006144 m²
"""
import pytest
import numpy as np
from fahts.core.model.section import BoxSection
from fahts.core.heat.section_mesh import BoxMesher, SectionMesh


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _sym_box() -> BoxSection:
    """200×200×8 mm symmetric BOX."""
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def _sym_mesh(elem_size: float = 0.04, n_layers: int = 1) -> SectionMesh:
    return BoxMesher(_sym_box(), elem_size=elem_size, n_layers=n_layers).build()


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_section_mesh(self):
        assert isinstance(_sym_mesh(), SectionMesh)

    def test_nodes_shape(self):
        m = _sym_mesh()
        assert m.nodes.ndim == 2 and m.nodes.shape[1] == 2

    def test_quads_shape(self):
        m = _sym_mesh()
        assert m.quads.ndim == 2 and m.quads.shape[1] == 4

    def test_outer_edge_pairs_shape(self):
        m = _sym_mesh()
        assert m.outer_edge_pairs.ndim == 2 and m.outer_edge_pairs.shape[1] == 2

    def test_inner_edge_pairs_shape(self):
        m = _sym_mesh()
        assert m.inner_edge_pairs.ndim == 2 and m.inner_edge_pairs.shape[1] == 2


# ── Node and element counts ───────────────────────────────────────────────────

class TestCounts:
    def test_n_quads(self):
        # bot(7) + top(7) + left_web(5) + right_web(5) = 24
        assert _sym_mesh().n_quads == 24

    def test_n_nodes(self):
        # 16 + 16 + 8 + 8 = 48
        assert _sym_mesh().n_nodes == 48

    def test_outer_edge_count(self):
        # 4 sides × 7 edges/side = 28
        assert len(_sym_mesh().outer_edge_pairs) == 28

    def test_inner_edge_count(self):
        # 4 hollow faces × 5 edges = 20
        assert len(_sym_mesh().inner_edge_pairs) == 20

    def test_n_layers_2_doubles_through_thickness(self):
        m1 = _sym_mesh(n_layers=1)
        m2 = _sym_mesh(n_layers=2)
        assert m2.n_quads > m1.n_quads

    def test_smaller_elem_size_gives_more_quads(self):
        m_coarse = _sym_mesh(elem_size=0.04)
        m_fine   = _sym_mesh(elem_size=0.02)
        assert m_fine.n_quads > m_coarse.n_quads


# ── Geometry correctness ──────────────────────────────────────────────────────

class TestGeometry:
    def test_outer_perimeter(self):
        m = _sym_mesh()
        assert abs(m.outer_perimeter - 0.8) < 1e-9

    def test_inner_perimeter(self):
        m = _sym_mesh()
        # 2*(0.184 + 0.184) = 0.736
        assert abs(m.inner_perimeter - 0.736) < 1e-9

    def test_steel_area(self):
        sec = _sym_box()
        m = _sym_mesh()
        expected = sec.cross_section_area  # = 0.006144 m²
        assert abs(m.steel_area - expected) < 1e-9

    def test_all_nodes_within_outer_bounds(self):
        sec = _sym_box()
        m = _sym_mesh()
        y, z = m.nodes[:, 0], m.nodes[:, 1]
        assert np.all(y >= -sec.W / 2 - 1e-10)
        assert np.all(y <=  sec.W / 2 + 1e-10)
        assert np.all(z >= -sec.H / 2 - 1e-10)
        assert np.all(z <=  sec.H / 2 + 1e-10)

    def test_no_nodes_in_hollow(self):
        sec = _sym_box()
        m = _sym_mesh()
        y, z = m.nodes[:, 0], m.nodes[:, 1]
        yl = -sec.W / 2 + sec.T_side
        yr =  sec.W / 2 - sec.T_side
        zb = -sec.H / 2 + sec.T_bot
        zt =  sec.H / 2 - sec.T_top
        tol = 1e-9
        in_hollow = (
            (y > yl + tol) & (y < yr - tol) &
            (z > zb + tol) & (z < zt - tol)
        )
        assert not np.any(in_hollow), "Nodes found inside hollow interior"

    def test_quad_node_indices_in_range(self):
        m = _sym_mesh()
        assert m.quads.min() >= 0
        assert m.quads.max() < m.n_nodes

    def test_outer_edge_indices_in_range(self):
        m = _sym_mesh()
        assert m.outer_edge_pairs.min() >= 0
        assert m.outer_edge_pairs.max() < m.n_nodes

    def test_inner_edge_indices_in_range(self):
        m = _sym_mesh()
        assert m.inner_edge_pairs.min() >= 0
        assert m.inner_edge_pairs.max() < m.n_nodes


# ── CCW orientation (positive Jacobian) ───────────────────────────────────────

class TestOrientation:
    def test_all_quads_ccw(self):
        """All quads must have positive signed area (CCW = positive in y-z plane)."""
        m = _sym_mesh()
        for quad in m.quads:
            y = m.nodes[quad, 0]
            z = m.nodes[quad, 1]
            # Signed area via shoelace for quad (CCW → positive)
            signed = 0.5 * (
                (y[0] * z[1] - y[1] * z[0])
                + (y[1] * z[2] - y[2] * z[1])
                + (y[2] * z[3] - y[3] * z[2])
                + (y[3] * z[0] - y[0] * z[3])
            )
            assert signed > 0, f"Quad {list(quad)} has non-positive signed area {signed}"


# ── Boundary edge locations ───────────────────────────────────────────────────

class TestBoundaryEdges:
    def test_outer_edges_on_outer_rectangle(self):
        sec = _sym_box()
        m = _sym_mesh()
        y0, y1 = -sec.W / 2, sec.W / 2
        z0, z1 = -sec.H / 2, sec.H / 2
        tol = 1e-9
        for a, b in m.outer_edge_pairs:
            ya, za = m.nodes[a]
            yb, zb = m.nodes[b]
            on_rect = (
                (abs(za - z0) < tol and abs(zb - z0) < tol)
                or (abs(za - z1) < tol and abs(zb - z1) < tol)
                or (abs(ya - y0) < tol and abs(yb - y0) < tol)
                or (abs(ya - y1) < tol and abs(yb - y1) < tol)
            )
            assert on_rect, f"Outer edge ({a},{b}) not on outer rectangle"

    def test_inner_edges_on_hollow_perimeter(self):
        sec = _sym_box()
        m = _sym_mesh()
        yl = -sec.W / 2 + sec.T_side
        yr =  sec.W / 2 - sec.T_side
        zb = -sec.H / 2 + sec.T_bot
        zt =  sec.H / 2 - sec.T_top
        tol = 1e-9
        for a, b in m.inner_edge_pairs:
            ya, za = m.nodes[a]
            yb, zb_n = m.nodes[b]
            on_inner = (
                (abs(za - zb) < tol and abs(zb_n - zb) < tol)
                or (abs(za - zt) < tol and abs(zb_n - zt) < tol)
                or (abs(ya - yl) < tol and abs(yb - yl) < tol)
                or (abs(ya - yr) < tol and abs(yb - yr) < tol)
            )
            assert on_inner, f"Inner edge ({a},{b}) not on hollow perimeter"

    def test_no_overlap_between_outer_and_inner_edges(self):
        m = _sym_mesh()
        outer_set = {tuple(sorted(e)) for e in m.outer_edge_pairs}
        inner_set = {tuple(sorted(e)) for e in m.inner_edge_pairs}
        assert outer_set.isdisjoint(inner_set)


# ── Asymmetric section ────────────────────────────────────────────────────────

class TestAsymmetricBox:
    def test_asymmetric_T_top_T_bot(self):
        sec = BoxSection(sid=2, H=0.30, W=0.15, T_side=0.006, T_bot=0.010, T_top=0.006)
        m = BoxMesher(sec, elem_size=0.03, n_layers=1).build()

        assert m.n_quads > 0
        assert m.n_nodes > 0

        expected_outer = 2 * (sec.H + sec.W)
        assert abs(m.outer_perimeter - expected_outer) < 1e-9

        expected_inner = 2 * (sec.inner_height + sec.inner_width)
        assert abs(m.inner_perimeter - expected_inner) < 1e-9

        assert abs(m.steel_area - sec.cross_section_area) < 1e-9

    def test_n_layers_2(self):
        sec = _sym_box()
        m = BoxMesher(sec, elem_size=0.04, n_layers=2).build()
        assert m.n_quads > 0
        # Area must still equal section area regardless of refinement
        assert abs(m.steel_area - sec.cross_section_area) < 1e-9
        # Outer/inner perimeters unchanged
        assert abs(m.outer_perimeter - 0.8) < 1e-9
        assert abs(m.inner_perimeter - 0.736) < 1e-9


# ── Default elem_size ─────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_elem_size_builds_successfully(self):
        sec = _sym_box()
        m = BoxMesher(sec).build()  # no elem_size → defaults to T_side=0.008
        assert m.n_quads > 0

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            BoxMesher(_sym_box(), n_layers=0)

    def test_zero_hollow_raises(self):
        # T_side >= W/2 → inner_width <= 0
        with pytest.raises(ValueError):
            BoxMesher(BoxSection(sid=99, H=0.2, W=0.2, T_side=0.1, T_bot=0.008, T_top=0.008))
