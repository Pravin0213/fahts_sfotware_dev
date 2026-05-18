"""Tests for fahts/gui/panels/results_panel.py (Phase 4.5 / 4.6)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")

from PyQt6.QtWidgets import QApplication  # noqa: E402
from fahts.gui.panels.results_panel import ResultsPanel  # noqa: E402
from fahts.core.model.section import BoxSection  # noqa: E402
from fahts.core.heat.section_mesh.box_mesher import BoxMesher  # noqa: E402


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _box_section() -> BoxSection:
    return BoxSection(sid=1, H=0.3, T_side=0.01, T_bot=0.01, T_top=0.01, W=0.2)


def _make_model(eid: int = 1, geom_id: int = 1, box: bool = True):
    """Minimal mock FEMModel with one element."""
    section = _box_section() if box else SimpleNamespace(sid=geom_id)
    elem = SimpleNamespace(eid=eid, n1=1, n2=2, mat_id=1, geom_id=geom_id)
    return SimpleNamespace(elements={eid: elem}, sections={geom_id: section})


def _make_result(eid: int = 1, n_steps: int = 5) -> SimpleNamespace:
    """Minimal mock TemperatureField with T_section and T_centroid data."""
    section = _box_section()
    mesh = BoxMesher(section).build()
    n_nodes = len(mesh.nodes)
    rng = np.random.default_rng(42)
    T_sec = rng.uniform(20.0, 700.0, (n_steps, n_nodes))
    times = np.linspace(0.0, 3600.0, n_steps)
    return SimpleNamespace(
        times=times,
        element_ids=[eid],
        T_centroid=rng.uniform(20.0, 700.0, (n_steps, 1)),
        T_section={eid: T_sec},
    )


@pytest.fixture
def panel(qapp):
    p = ResultsPanel()
    yield p
    p.close()


@pytest.fixture
def loaded_panel(qapp):
    """Panel after show_section() with valid BOX element and results."""
    p = ResultsPanel()
    model = _make_model()
    result = _make_result()
    p.show_section(1, model, result, t_idx=0)
    yield p
    p.close()


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_panel_created(self, panel):
        assert panel is not None

    def test_initial_eid_is_none(self, panel):
        assert panel._eid is None

    def test_initial_result_is_none(self, panel):
        assert panel._result is None

    def test_initial_placeholder_visible(self, panel):
        assert not panel._placeholder.isHidden()

    def test_initial_canvas_hidden(self, panel):
        assert panel._canvas.isHidden()

    def test_initial_triang_is_none(self, panel):
        assert panel._triang is None

    def test_initial_mesh_nodes_is_none(self, panel):
        assert panel._mesh_nodes is None

    # Task 4.6 widgets
    def test_tt_placeholder_exists(self, panel):
        assert hasattr(panel, "_tt_placeholder")

    def test_tt_canvas_exists(self, panel):
        assert hasattr(panel, "_tt_canvas")

    def test_tt_placeholder_visible_initially(self, panel):
        assert not panel._tt_placeholder.isHidden()

    def test_tt_canvas_hidden_initially(self, panel):
        assert panel._tt_canvas.isHidden()

    def test_tt_vline_none_initially(self, panel):
        assert panel._tt_vline is None


# ── clear() ───────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_resets_eid(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._eid is None

    def test_clear_resets_result(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._result is None

    def test_clear_hides_canvas(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._canvas.isHidden()

    def test_clear_shows_placeholder(self, loaded_panel):
        loaded_panel.clear()
        assert not loaded_panel._placeholder.isHidden()

    def test_clear_resets_triang(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._triang is None

    def test_clear_hides_tt_canvas(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._tt_canvas.isHidden()

    def test_clear_shows_tt_placeholder(self, loaded_panel):
        loaded_panel.clear()
        assert not loaded_panel._tt_placeholder.isHidden()

    def test_clear_resets_tt_vline(self, loaded_panel):
        loaded_panel.clear()
        assert loaded_panel._tt_vline is None


# ── show_section() — cross-section ────────────────────────────────────────────

class TestShowSection:
    def test_show_sets_eid(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result, t_idx=0)
        assert panel._eid == 1

    def test_show_sets_result(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._result is result

    def test_show_sets_t_idx(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result, t_idx=3)
        assert panel._t_idx == 3

    def test_show_builds_triang(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._triang is not None

    def test_show_builds_mesh_nodes(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._mesh_nodes is not None
        assert panel._mesh_nodes.shape[1] == 2

    def test_canvas_shown_after_show(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert not panel._canvas.isHidden()

    def test_placeholder_hidden_after_show(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._placeholder.isHidden()

    def test_show_non_box_element_hides_canvas(self, panel):
        """Non-BOX section → cross-section canvas hidden (no mesh)."""
        model = _make_model(box=False)
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._canvas.isHidden()

    def test_show_eid_not_in_T_section_hides_canvas(self, panel):
        """eid 99 has no T_section data → cross-section canvas hidden."""
        model = _make_model(eid=99)
        result = _make_result(eid=1)
        panel.show_section(99, model, result)
        assert panel._canvas.isHidden()

    def test_show_unknown_eid_hides_canvas(self, panel):
        """eid not in model.elements → canvas hidden."""
        model = _make_model(eid=1)
        result = _make_result(eid=99)
        panel.show_section(99, model, result)
        assert panel._canvas.isHidden()

    def test_show_twice_replaces_plot(self, panel):
        model = _make_model()
        result1 = _make_result(n_steps=3)
        result2 = _make_result(n_steps=7)
        panel.show_section(1, model, result1)
        panel.show_section(1, model, result2)
        assert panel._result is result2
        assert not panel._canvas.isHidden()


# ── show_section() — T-t graph (Task 4.6) ────────────────────────────────────

class TestShowSectionTT:
    def test_tt_canvas_shown_after_show(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert not panel._tt_canvas.isHidden()

    def test_tt_placeholder_hidden_after_show(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._tt_placeholder.isHidden()

    def test_tt_vline_set_after_show(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        assert panel._tt_vline is not None

    def test_tt_vline_at_correct_time(self, panel):
        model = _make_model()
        result = _make_result(n_steps=5)
        panel.show_section(1, model, result, t_idx=2)
        expected_t = float(result.times[2])
        assert abs(panel._tt_vline.get_xdata()[0] - expected_t) < 1e-6

    def test_tt_canvas_hidden_when_eid_not_in_element_ids(self, panel):
        """eid in T_section but not in element_ids → T-t canvas hidden."""
        model = _make_model(eid=1)
        result = _make_result(eid=1)
        # Override element_ids to exclude eid=1
        result.element_ids = [99]
        panel.show_section(1, model, result)
        assert panel._tt_canvas.isHidden()

    def test_tt_canvas_hidden_for_non_box(self, panel):
        """Non-BOX section: T-t canvas can still show if eid in element_ids."""
        model = _make_model(box=False)
        result = _make_result()
        # element_ids = [1], T_centroid has data — T-t should show
        panel.show_section(1, model, result)
        assert not panel._tt_canvas.isHidden()

    def test_tt_shows_even_if_cross_section_hidden(self, panel):
        """T-t graph is independent of cross-section availability."""
        model = _make_model(box=False)   # no BOX → cross-section hidden
        result = _make_result()           # has T_centroid → T-t shown
        panel.show_section(1, model, result)
        assert panel._canvas.isHidden()
        assert not panel._tt_canvas.isHidden()


# ── update_time() ─────────────────────────────────────────────────────────────

class TestUpdateTime:
    def test_update_changes_t_idx(self, loaded_panel):
        loaded_panel.update_time(3)
        assert loaded_panel._t_idx == 3

    def test_update_does_not_crash_when_no_result(self, panel):
        panel.update_time(2)   # no result loaded — must not raise
        assert panel._t_idx == 2

    def test_update_clamped_t_idx_beyond_steps(self, loaded_panel):
        """Overshooting the last step must not raise."""
        loaded_panel.update_time(999)
        assert not loaded_panel._canvas.isHidden()

    def test_update_keeps_canvas_visible(self, loaded_panel):
        loaded_panel.update_time(2)
        assert not loaded_panel._canvas.isHidden()

    def test_update_moves_tt_vline(self, loaded_panel):
        """update_time repositions the marker without a full replot."""
        result = loaded_panel._result
        loaded_panel.update_time(3)
        expected_t = float(result.times[3])
        x = loaded_panel._tt_vline.get_xdata()[0]
        assert abs(x - expected_t) < 1e-6

    def test_update_tt_vline_at_step_zero(self, loaded_panel):
        loaded_panel.update_time(0)
        expected_t = float(loaded_panel._result.times[0])
        assert abs(loaded_panel._tt_vline.get_xdata()[0] - expected_t) < 1e-6

    def test_update_tt_vline_at_last_step(self, loaded_panel):
        n = len(loaded_panel._result.times)
        loaded_panel.update_time(n - 1)
        expected_t = float(loaded_panel._result.times[n - 1])
        assert abs(loaded_panel._tt_vline.get_xdata()[0] - expected_t) < 1e-6

    def test_update_keeps_tt_canvas_visible(self, loaded_panel):
        loaded_panel.update_time(2)
        assert not loaded_panel._tt_canvas.isHidden()


# ── Triangulation ─────────────────────────────────────────────────────────────

class TestTriangulation:
    def test_triang_y_matches_mesh_nodes(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        nodes = panel._mesh_nodes
        np.testing.assert_array_almost_equal(panel._triang.x, nodes[:, 0])

    def test_triang_z_matches_mesh_nodes(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        nodes = panel._mesh_nodes
        np.testing.assert_array_almost_equal(panel._triang.y, nodes[:, 1])

    def test_triang_triangles_count(self, panel):
        model = _make_model()
        result = _make_result()
        panel.show_section(1, model, result)
        section = _box_section()
        mesh = BoxMesher(section).build()
        expected = 2 * len(mesh.quads)
        assert len(panel._triang.triangles) == expected
