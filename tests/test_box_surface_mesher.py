"""
Tests for BoxSurfaceMesher and BeamSurfaceMesh.

Covers mesh structure, node deduplication, element counts, thickness
assignments, and local_cols correctness.
"""
import math
import numpy as np
import pytest

from fahts.core.model.section import BoxSection
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh


# ── Fixture ────────────────────────────────────────────────────────────────────

def _box() -> BoxSection:
    """300×300×10 mm BOX section."""
    return BoxSection(sid=1, H=0.300, T_side=0.010, T_bot=0.010, T_top=0.010, W=0.300)


def _wide_box() -> BoxSection:
    """400 mm wide × 200 mm high, unequal flanges."""
    return BoxSection(sid=2, H=0.200, T_side=0.008, T_bot=0.012, T_top=0.010, W=0.400)


# ── Node / quad counts ────────────────────────────────────────────────────────

class TestDefaultMesh:
    """Default n_top=2, n_side=3, n_length=4."""

    def setup_method(self):
        self.mesh = BoxSurfaceMesher(_box(), length=2.0).build()

    def test_returns_beam_surface_mesh(self):
        assert isinstance(self.mesh, BeamSurfaceMesh)

    def test_quad_count(self):
        # 4 × (2×n_top + 2×n_side) elements
        # n_length × (2×2 + 2×3) = 4 × 10 = 40
        assert self.mesh.n_quads == 40

    def test_node_count(self):
        # 2(n_length+1)(n_top + n_side) = 2 × 5 × 5 = 50
        assert self.mesh.n_nodes == 50

    def test_thicknesses_shape(self):
        assert self.mesh.thicknesses.shape == (40,)

    def test_local_cols_shape(self):
        assert self.mesh.local_cols.shape == (40, 2)

    def test_quads_shape(self):
        assert self.mesh.quads.shape == (40, 4)

    def test_nodes_shape(self):
        assert self.mesh.nodes.shape == (50, 3)

    def test_all_node_indices_in_range(self):
        assert self.mesh.quads.min() >= 0
        assert self.mesh.quads.max() < self.mesh.n_nodes

    def test_unique_quads(self):
        """No two quads share all four nodes."""
        seen = set()
        for quad in self.mesh.quads:
            key = tuple(sorted(quad))
            assert key not in seen, f"Duplicate quad: {key}"
            seen.add(key)


class TestCustomMesh:
    """Custom n_top=1, n_side=1, n_length=2."""

    def setup_method(self):
        self.mesh = BoxSurfaceMesher(_box(), length=1.0, n_top=1, n_side=1, n_length=2).build()

    def test_quad_count(self):
        # 2 × (2×1 + 2×1) = 2 × 4 = 8
        assert self.mesh.n_quads == 8

    def test_node_count(self):
        # 2(n_length+1)(n_top + n_side) = 2 × 3 × 2 = 12
        assert self.mesh.n_nodes == 12


# ── Coordinate correctness ────────────────────────────────────────────────────

class TestCoordinates:

    def setup_method(self):
        self.sec   = _box()
        self.L     = 3.0
        self.mesh  = BoxSurfaceMesher(self.sec, length=self.L).build()
        self.nodes = self.mesh.nodes

    def test_x_range(self):
        xs = self.nodes[:, 0]
        assert float(xs.min()) == pytest.approx(0.0)
        assert float(xs.max()) == pytest.approx(self.L)

    def test_y_range(self):
        ys = self.nodes[:, 1]
        hy = self.sec.W / 2.0
        assert float(ys.min()) == pytest.approx(-hy)
        assert float(ys.max()) == pytest.approx(+hy)

    def test_z_range(self):
        zs = self.nodes[:, 2]
        hz = self.sec.H / 2.0
        assert float(zs.min()) == pytest.approx(-hz)
        assert float(zs.max()) == pytest.approx(+hz)

    def test_bottom_face_z(self):
        """All bottom-face elements have z = -H/2 for all nodes."""
        hz = self.sec.H / 2.0
        bottom_quads = self.mesh.quads[:self.mesh.n_quads // 2][:2 * 4]  # first n_top*n_length quads
        # Identify bottom-face quads by local_cols == [0, 1] and z = -hz
        for q, quad in enumerate(self.mesh.quads):
            if list(self.mesh.local_cols[q]) == [0, 1]:
                zs = self.nodes[quad, 2]
                if np.allclose(zs, -hz, atol=1e-9):
                    # bottom face — check thicknesses
                    assert float(self.mesh.thicknesses[q]) == pytest.approx(self.sec.T_bot)

    def test_top_face_thickness(self):
        hz = self.sec.H / 2.0
        for q, quad in enumerate(self.mesh.quads):
            if list(self.mesh.local_cols[q]) == [0, 1]:
                zs = self.nodes[quad, 2]
                if np.allclose(zs, +hz, atol=1e-9):
                    assert float(self.mesh.thicknesses[q]) == pytest.approx(self.sec.T_top)

    def test_side_face_thickness(self):
        for q, quad in enumerate(self.mesh.quads):
            if list(self.mesh.local_cols[q]) == [0, 2]:
                assert float(self.mesh.thicknesses[q]) == pytest.approx(self.sec.T_side)


# ── Thickness assignments ─────────────────────────────────────────────────────

class TestThicknesses:

    def test_unequal_flanges(self):
        """Different T_bot, T_top, T_side all appear correctly."""
        sec  = _wide_box()
        mesh = BoxSurfaceMesher(sec, length=2.0).build()
        hz   = sec.H / 2.0

        T_bot_seen  = set()
        T_top_seen  = set()
        T_side_seen = set()

        for q, quad in enumerate(mesh.quads):
            t = float(mesh.thicknesses[q])
            if list(mesh.local_cols[q]) == [0, 1]:
                zs = mesh.nodes[quad, 2]
                if np.allclose(zs, -hz, atol=1e-9):
                    T_bot_seen.add(t)
                elif np.allclose(zs, +hz, atol=1e-9):
                    T_top_seen.add(t)
            elif list(mesh.local_cols[q]) == [0, 2]:
                T_side_seen.add(t)

        assert len(T_bot_seen)  == 1 and next(iter(T_bot_seen))  == pytest.approx(sec.T_bot)
        assert len(T_top_seen)  == 1 and next(iter(T_top_seen))  == pytest.approx(sec.T_top)
        assert len(T_side_seen) == 1 and next(iter(T_side_seen)) == pytest.approx(sec.T_side)


# ── Node deduplication ────────────────────────────────────────────────────────

class TestDeduplication:

    def test_no_duplicate_nodes(self):
        mesh = BoxSurfaceMesher(_box(), length=2.0).build()
        coords_set = set()
        for node in mesh.nodes:
            key = (round(node[0], 9), round(node[1], 9), round(node[2], 9))
            assert key not in coords_set, f"Duplicate node at {key}"
            coords_set.add(key)

    def test_corner_nodes_shared(self):
        """
        Each corner of the BOX cross-section (4 corners × n_length+1 x-positions)
        must appear as a single node shared by adjacent faces.
        """
        sec  = _box()
        L    = 2.0
        mesh = BoxSurfaceMesher(sec, length=L).build()

        hy = sec.W / 2.0
        hz = sec.H / 2.0
        xs = np.linspace(0.0, L, 5)  # n_length=4 → 5 positions

        for xi in xs:
            for yc, zc in [(-hy, -hz), (+hy, -hz), (-hy, +hz), (+hy, +hz)]:
                matches = [
                    i for i, n in enumerate(mesh.nodes)
                    if (abs(n[0] - xi) < 1e-9 and abs(n[1] - yc) < 1e-9
                        and abs(n[2] - zc) < 1e-9)
                ]
                assert len(matches) == 1, (
                    f"Corner ({xi:.2f},{yc},{zc}) appears {len(matches)} times"
                )


# ── Area weights ──────────────────────────────────────────────────────────────

class TestAreaWeights:

    def test_weights_sum_to_one(self):
        mesh = BoxSurfaceMesher(_box(), length=2.0).build()
        w = mesh.node_area_weights
        assert float(w.sum()) == pytest.approx(1.0)

    def test_weights_nonnegative(self):
        mesh = BoxSurfaceMesher(_box(), length=2.0).build()
        assert (mesh.node_area_weights >= 0.0).all()

    def test_total_surface_area(self):
        """Area weights × node count should reflect total surface area."""
        sec  = _box()
        L    = 2.0
        mesh = BoxSurfaceMesher(sec, length=L).build()

        # Compute total face area from element areas
        total = 0.0
        for q, quad in enumerate(mesh.quads):
            coords = mesh.nodes[quad][:, mesh.local_cols[q]]
            d1 = coords[2] - coords[0]
            d2 = coords[3] - coords[1]
            total += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])

        expected = (2 * sec.W + 2 * sec.H) * L
        assert total == pytest.approx(expected, rel=1e-10)


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="length must be positive"):
            BoxSurfaceMesher(_box(), length=0.0)

    def test_negative_n_top_raises(self):
        with pytest.raises(ValueError):
            BoxSurfaceMesher(_box(), length=1.0, n_top=0)

    def test_negative_n_side_raises(self):
        with pytest.raises(ValueError):
            BoxSurfaceMesher(_box(), length=1.0, n_side=0)

    def test_negative_n_length_raises(self):
        with pytest.raises(ValueError):
            BoxSurfaceMesher(_box(), length=1.0, n_length=0)
