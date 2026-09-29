"""Tests for fahts/gui/panels/properties_panel.py (Phase 1.8)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402
from fahts.gui.panels.properties_panel import PropertiesPanel  # noqa: E402


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    import sys
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def real_model(qapp):
    from fahts.core.io.usfos_reader import read_usfos_fem
    p = Path(__file__).parents[1] / "examples" / "models" / "model_file.fem"
    if not p.exists():
        pytest.skip("model_file.fem not found")
    return read_usfos_fem(p)


@pytest.fixture
def panel(qapp):
    p = PropertiesPanel()
    p.show()
    qapp.processEvents()
    yield p
    p.close()


@pytest.fixture
def panel_loaded(panel, real_model):
    first_eid = sorted(real_model.elements.keys())[0]
    panel.show_element(first_eid, real_model)
    return panel, first_eid


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_panel_created(self, panel):
        assert panel is not None

    def test_current_eid_none_initially(self, panel):
        assert panel.current_eid is None

    def test_placeholder_visible_initially(self, panel):
        assert panel._placeholder.isVisible()

    def test_scroll_area_hidden_initially(self, panel):
        assert not panel._scroll_area.isVisible()


# ── show_element ──────────────────────────────────────────────────────────────

class TestShowElement:
    def test_no_crash(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)

    def test_current_eid_set(self, panel_loaded):
        panel, eid = panel_loaded
        assert panel.current_eid == eid

    def test_placeholder_hidden(self, panel_loaded):
        panel, _ = panel_loaded
        assert not panel._placeholder.isVisible()

    def test_scroll_area_visible(self, panel_loaded):
        panel, _ = panel_loaded
        assert panel._scroll_area.isVisible()

    def test_eid_shown_in_content(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        text = _collect_labels(panel)
        assert str(eid) in text

    def test_length_shown_in_content(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        elem = real_model.elements[eid]
        text = _collect_labels(panel)
        assert f"{elem.length:.3f}" in text

    def test_node_ids_shown(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        elem = real_model.elements[eid]
        text = _collect_labels(panel)
        assert str(elem.n1) in text
        assert str(elem.n2) in text

    def test_section_dimensions_shown(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        text = _collect_labels(panel)
        assert "mm" in text   # at least one dimension in mm

    def test_material_properties_shown(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        text = _collect_labels(panel)
        assert "GPa" in text
        assert "MPa" in text

    def test_group_membership_shown(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        groups_for_eid = [
            name for name, g in real_model.groups.items()
            if eid in g.element_ids
        ]
        if not groups_for_eid:
            pytest.skip("first element is not in any group")
        panel.show_element(eid, real_model)
        text = _collect_labels(panel)
        assert groups_for_eid[0] in text

    def test_element_selected_signal_emitted(self, panel, real_model, qapp):
        eid = sorted(real_model.elements.keys())[0]
        received = []
        panel.element_selected.connect(received.append)
        panel.show_element(eid, real_model)
        assert received == [eid]

    def test_repopulate_replaces_content(self, panel, real_model):
        eids = sorted(real_model.elements.keys())
        panel.show_element(eids[0], real_model)
        panel.show_element(eids[1], real_model)
        assert panel.current_eid == eids[1]
        text = _collect_labels(panel)
        assert str(eids[1]) in text

    def test_unknown_eid_calls_clear(self, panel, real_model):
        panel.show_element(99999999, real_model)
        assert panel.current_eid is None
        assert panel._placeholder.isVisible()

    def test_all_elements_no_crash(self, panel, real_model):
        for eid in real_model.elements:
            panel.show_element(eid, real_model)


# ── clear ─────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_after_show(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        panel.clear()
        assert panel.current_eid is None

    def test_placeholder_visible_after_clear(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        panel.clear()
        assert panel._placeholder.isVisible()

    def test_scroll_hidden_after_clear(self, panel, real_model):
        eid = sorted(real_model.elements.keys())[0]
        panel.show_element(eid, real_model)
        panel.clear()
        assert not panel._scroll_area.isVisible()

    def test_clear_when_already_clear(self, panel):
        panel.clear()   # must not crash on double-clear


# ── SceneManager picking integration ─────────────────────────────────────────

class TestPickingIntegration:
    def test_enable_picking_no_crash(self, real_model):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        received: list[int] = []
        scene.enable_picking(received.append)
        scene.disable_picking()
        scene.close()

    def test_on_cell_picked_calls_callback(self, real_model):
        """Simulate a picked mesh carrying element_id cell data."""
        import numpy as np
        import pyvista as pv
        from fahts.renderer.scene_manager import SceneManager

        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)

        received: list[int] = []
        scene.enable_picking(received.append)

        # One-triangle mesh so cell_data can be set
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        fake_picked = pv.PolyData(pts, np.array([3, 0, 1, 2]))
        fake_picked.cell_data["element_id"] = np.array([42], dtype=np.int32)
        scene._on_cell_picked(fake_picked)

        assert received == [42]
        scene.close()

    def test_on_cell_picked_none_is_noop(self, real_model):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        received: list[int] = []
        scene.enable_picking(received.append)
        scene._on_cell_picked(None)
        assert received == []
        scene.close()

    def test_on_cell_picked_no_element_id_is_noop(self, real_model):
        import numpy as np
        import pyvista as pv
        from fahts.renderer.scene_manager import SceneManager

        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        received: list[int] = []
        scene.enable_picking(received.append)

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        fake = pv.PolyData(pts, np.array([3, 0, 1, 2]))
        fake.cell_data["other_data"] = np.array([99], dtype=np.int32)
        scene._on_cell_picked(fake)
        assert received == []
        scene.close()

    def test_disable_picking_no_crash(self, real_model):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        scene.enable_picking(lambda eid: None)
        scene.disable_picking()
        scene.disable_picking()   # double-disable must not crash
        scene.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_labels(panel: PropertiesPanel) -> str:
    """Collect all visible text from the content widget labels."""
    from PyQt6.QtWidgets import QLabel
    parts: list[str] = []
    for lbl in panel._content_widget.findChildren(QLabel):
        parts.append(lbl.text())
    return " ".join(parts)
