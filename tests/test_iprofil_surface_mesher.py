"""
Tests for IProfileSurfaceMesher and its BeamSurfaceMesh output.

Covers mesh structure, node deduplication, junction connectivity, element counts,
thickness assignments, and area weights.
"""
import numpy as np
import pytest

from fahts.core.model.section import ISection
from fahts.core.heat.section_mesh.iprofil_surface_mesher import IProfileSurfaceMesher
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _isec() -> ISection:
    """Standard HEB 200 -like section (dimensions in metres)."""
    return ISection(
        sid=1,
        h=0.200,       # total height
        tw=0.009,      # web thickness
        bf_top=0.200,  # top flange width
        tf_top=0.015,  # top flange thickness
        bf_bot=0.200,  # bottom flange width
        tf_bot=0.015,  # bottom flange thickness
    )


def _narrow_isec() -> ISection:
    """Asymmetric I-section with different top/bottom flanges."""
    return ISection(
        sid=2,
        h=0.300,
        tw=0.010,
        bf_top=0.150,
        tf_top=0.012,
        bf_bot=0.100,
        tf_bot=0.010,
    )


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_beam_surface_mesh(self):
        mesh = IProfileSurfaceMesher(_isec(), length=2.0).build()
        assert isinstance(mesh, BeamSurfaceMesh)


# ── Mesh counts ───────────────────────────────────────────────────────────────

class TestDefaultMeshCounts:
    """Default n_top=4, n_side=2, n_bottom=2, n_length=2."""

    def setup_method(self):
        self.sec = _isec()
        self.mesh = IProfileSurfaceMesher(self.sec, length=2.0).build()

    def test_quad_count_positive(self):
        assert self.mesh.n_quads > 0

    def test_node_count_positive(self):
        assert self.mesh.n_nodes > 0

    def test_thicknesses_shape(self):
        assert self.mesh.thicknesses.shape == (self.mesh.n_quads,)

    def test_local_cols_shape(self):
        assert self.mesh.local_cols.shape == (self.mesh.n_quads, 2)

    def test_quads_shape(self):
        assert self.mesh.quads.shape == (self.mesh.n_quads, 4)

    def test_nodes_shape(self):
        assert self.mesh.nodes.shape == (self.mesh.n_nodes, 3)

    def test_all_node_indices_in_range(self):
        assert self.mesh.quads.min() >= 0
        assert self.mesh.quads.max() < self.mesh.n_nodes


class TestCustomMeshCounts:
    """Custom n_top=2, n_side=1, n_bottom=1, n_length=3."""

    def setup_method(self):
        self.mesh = IProfileSurfaceMesher(
            _isec(), length=1.0, n_top=2, n_side=1, n_bottom=1, n_length=3
        ).build()

    def test_quad_count_positive(self):
        assert self.mesh.n_quads > 0

    def test_all_node_indices_in_range(self):
        assert self.mesh.quads.min() >= 0
        assert self.mesh.quads.max() < self.mesh.n_nodes


# ── Coordinate ranges ─────────────────────────────────────────────────────────

class TestCoordinateRanges:

    def setup_method(self):
        self.sec  = _isec()
        self.L    = 3.0
        self.mesh = IProfileSurfaceMesher(self.sec, length=self.L).build()

    def test_x_range(self):
        xs = self.mesh.nodes[:, 0]
        assert float(xs.min()) == pytest.approx(0.0)
        assert float(xs.max()) == pytest.approx(self.L)

    def test_y_range_within_flanges(self):
        ys = self.mesh.nodes[:, 1]
        half_bf = max(self.sec.bf_top, self.sec.bf_bot) / 2.0
        assert float(ys.min()) >= -half_bf - 1e-9
        assert float(ys.max()) <= +half_bf + 1e-9

    def test_z_range(self):
        # Flanges are meshed at their inner faces; web spans between them.
        z_top_in = self.sec.h / 2.0 - self.sec.tf_top
        z_bot_in = -self.sec.h / 2.0 + self.sec.tf_bot
        zs = self.mesh.nodes[:, 2]
        assert float(zs.min()) == pytest.approx(z_bot_in)
        assert float(zs.max()) == pytest.approx(z_top_in)


# ── Thickness assignments ─────────────────────────────────────────────────────

class TestThicknessAssignments:

    def test_top_flange_inner_face_thickness(self):
        """Quads on the top flange inner face (z = z_top_in) must have thickness = tf_top."""
        sec  = _isec()
        mesh = IProfileSurfaceMesher(sec, length=2.0).build()
        z_top_in = sec.h / 2.0 - sec.tf_top
        for q, quad in enumerate(mesh.quads):
            if np.allclose(mesh.nodes[quad, 2], z_top_in, atol=1e-9):
                assert float(mesh.thicknesses[q]) == pytest.approx(sec.tf_top)

    def test_bottom_flange_inner_face_thickness(self):
        """Quads on the bot flange inner face (z = z_bot_in) must have thickness = tf_bot."""
        sec  = _isec()
        mesh = IProfileSurfaceMesher(sec, length=2.0).build()
        z_bot_in = -sec.h / 2.0 + sec.tf_bot
        for q, quad in enumerate(mesh.quads):
            if np.allclose(mesh.nodes[quad, 2], z_bot_in, atol=1e-9):
                assert float(mesh.thicknesses[q]) == pytest.approx(sec.tf_bot)

    def test_web_face_thickness(self):
        """Quads on the web left face (all node |y| = tw/2) must have thickness = tw."""
        sec  = _isec()
        mesh = IProfileSurfaceMesher(sec, length=2.0).build()
        yw = sec.tw / 2.0
        for q, quad in enumerate(mesh.quads):
            ys = mesh.nodes[quad, 1]
            if np.allclose(np.abs(ys), yw, atol=1e-9):
                assert float(mesh.thicknesses[q]) == pytest.approx(sec.tw)

    def test_asymmetric_flanges(self):
        """Different tf_top, tf_bot, tw all appear correctly."""
        sec  = _narrow_isec()
        mesh = IProfileSurfaceMesher(sec, length=2.0).build()
        z_top_in = sec.h / 2.0 - sec.tf_top
        z_bot_in = -sec.h / 2.0 + sec.tf_bot
        yw       = sec.tw / 2.0

        top_ts   = set()
        bot_ts   = set()
        web_ts   = set()

        for q, quad in enumerate(mesh.quads):
            zs = mesh.nodes[quad, 2]
            ys = mesh.nodes[quad, 1]
            if np.allclose(zs, z_top_in, atol=1e-9):
                top_ts.add(round(float(mesh.thicknesses[q]), 9))
            if np.allclose(zs, z_bot_in, atol=1e-9):
                bot_ts.add(round(float(mesh.thicknesses[q]), 9))
            if np.allclose(np.abs(ys), yw, atol=1e-9):
                web_ts.add(round(float(mesh.thicknesses[q]), 9))

        assert len(top_ts) == 1
        assert next(iter(top_ts)) == pytest.approx(sec.tf_top)
        assert len(bot_ts) == 1
        assert next(iter(bot_ts)) == pytest.approx(sec.tf_bot)
        assert len(web_ts) == 1
        assert next(iter(web_ts)) == pytest.approx(sec.tw)


# ── Node deduplication ────────────────────────────────────────────────────────

class TestDeduplication:

    def test_no_duplicate_nodes(self):
        mesh = IProfileSurfaceMesher(_isec(), length=2.0).build()
        coords_set = set()
        for node in mesh.nodes:
            key = (round(node[0], 9), round(node[1], 9), round(node[2], 9))
            assert key not in coords_set, f"Duplicate node at {key}"
            coords_set.add(key)

    def test_unique_quads(self):
        mesh = IProfileSurfaceMesher(_isec(), length=2.0).build()
        seen = set()
        for quad in mesh.quads:
            key = tuple(sorted(quad))
            assert key not in seen, f"Duplicate quad {key}"
            seen.add(key)

    def test_web_flange_junction_nodes_shared(self):
        """
        The web left-face corner nodes at y = −tw/2 must be shared with the
        inner-flange face nodes at the same (x, y=-tw/2, z_top_in/z_bot_in).
        This verifies the T-junction FEM connectivity that allows heat conduction
        between web and flanges.
        """
        sec  = _isec()
        L    = 2.0
        nl   = 2  # n_length
        mesh = IProfileSurfaceMesher(sec, length=L, n_length=nl).build()

        z_top_in = sec.h / 2.0 - sec.tf_top
        z_bot_in = -sec.h / 2.0 + sec.tf_bot
        yw_l     = -sec.tw / 2.0   # web left-face y

        # web quads: all 4 nodes at |y| = tw/2 (left web face)
        # flange quads: all 4 nodes at z = z_top_in or z = z_bot_in
        web_node_ids = set()
        top_node_ids = set()
        bot_node_ids = set()
        for q, quad in enumerate(mesh.quads):
            ys = mesh.nodes[quad, 1]
            zs = mesh.nodes[quad, 2]
            if np.allclose(np.abs(ys), sec.tw / 2.0, atol=1e-9):
                web_node_ids.update(int(n) for n in quad)
            if np.allclose(zs, z_top_in, atol=1e-9):
                top_node_ids.update(int(n) for n in quad)
            if np.allclose(zs, z_bot_in, atol=1e-9):
                bot_node_ids.update(int(n) for n in quad)

        shared_top = web_node_ids & top_node_ids
        shared_bot = web_node_ids & bot_node_ids

        assert len(shared_top) >= nl + 1, (
            f"Expected ≥{nl+1} shared web↔top-flange nodes, got {len(shared_top)}"
        )
        assert len(shared_bot) >= nl + 1, (
            f"Expected ≥{nl+1} shared web↔bot-flange nodes, got {len(shared_bot)}"
        )

        # Each shared node must sit at y = yw_l and z = junction z
        for nid in shared_top:
            n = mesh.nodes[nid]
            assert abs(n[1] - yw_l) < 1e-9, f"Top junction node y={n[1]} ≠ {yw_l}"
            assert abs(n[2] - z_top_in) < 1e-9, f"Top junction node z={n[2]} ≠ {z_top_in}"
        for nid in shared_bot:
            n = mesh.nodes[nid]
            assert abs(n[1] - yw_l) < 1e-9, f"Bot junction node y={n[1]} ≠ {yw_l}"
            assert abs(n[2] - z_bot_in) < 1e-9, f"Bot junction node z={n[2]} ≠ {z_bot_in}"


# ── Area weights ──────────────────────────────────────────────────────────────

class TestAreaWeights:

    def test_weights_sum_to_one(self):
        mesh = IProfileSurfaceMesher(_isec(), length=2.0).build()
        assert float(mesh.node_area_weights.sum()) == pytest.approx(1.0)

    def test_weights_nonnegative(self):
        mesh = IProfileSurfaceMesher(_isec(), length=2.0).build()
        assert (mesh.node_area_weights >= 0.0).all()


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidation:

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="length must be positive"):
            IProfileSurfaceMesher(_isec(), length=0.0)

    def test_zero_n_top_raises(self):
        with pytest.raises(ValueError):
            IProfileSurfaceMesher(_isec(), length=1.0, n_top=0)

    def test_zero_n_side_raises(self):
        with pytest.raises(ValueError):
            IProfileSurfaceMesher(_isec(), length=1.0, n_side=0)

    def test_zero_n_bottom_raises(self):
        with pytest.raises(ValueError):
            IProfileSurfaceMesher(_isec(), length=1.0, n_bottom=0)

    def test_zero_n_length_raises(self):
        with pytest.raises(ValueError):
            IProfileSurfaceMesher(_isec(), length=1.0, n_length=0)
