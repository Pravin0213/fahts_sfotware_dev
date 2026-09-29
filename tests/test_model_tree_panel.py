"""Tests for fahts/gui/panels/model_tree_panel.py (Phase 1.7)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
from fahts.gui.panels.model_tree_panel import ModelTreePanel  # noqa: E402


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
    p = ModelTreePanel()
    yield p


@pytest.fixture
def panel_loaded(panel, real_model):
    panel.populate(real_model)
    return panel


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_has_tree(self, panel):
        assert panel._tree is not None

    def test_buttons_disabled_before_populate(self, panel):
        assert not panel._btn_show_all.isEnabled()
        assert not panel._btn_hide_all.isEnabled()

    def test_no_scene_initially(self, panel):
        assert panel._scene is None

    def test_set_scene(self, panel):
        mock_scene = MagicMock()
        panel.set_scene(mock_scene)
        assert panel._scene is mock_scene


# ── populate ─────────────────────────────────────────────────────────────────

class TestPopulate:
    def test_populate_no_crash(self, panel, real_model):
        panel.populate(real_model)

    def test_buttons_enabled_after_populate(self, panel_loaded):
        assert panel_loaded._btn_show_all.isEnabled()
        assert panel_loaded._btn_hide_all.isEnabled()

    def test_has_structure_root(self, panel_loaded):
        root = panel_loaded._tree.invisibleRootItem()
        labels = [root.child(i).text(0) for i in range(root.childCount())]
        assert any("Structure" in lbl for lbl in labels)

    def test_has_materials_root(self, panel_loaded):
        root = panel_loaded._tree.invisibleRootItem()
        labels = [root.child(i).text(0) for i in range(root.childCount())]
        assert any("Materials" in lbl for lbl in labels)

    def test_has_heat_sources_root(self, panel_loaded):
        root = panel_loaded._tree.invisibleRootItem()
        labels = [root.child(i).text(0) for i in range(root.childCount())]
        assert any("Heat Sources" in lbl for lbl in labels)

    def test_group_items_count(self, panel_loaded, real_model):
        checked = panel_loaded.checked_groups()
        assert len(checked) == len(real_model.groups)

    def test_all_groups_checked_initially(self, panel_loaded, real_model):
        checked = set(panel_loaded.checked_groups())
        expected = set(real_model.groups.keys())
        assert checked == expected

    def test_material_items_present(self, panel_loaded, real_model):
        # Find Materials root and count children
        root = panel_loaded._tree.invisibleRootItem()
        mat_root = None
        for i in range(root.childCount()):
            if "Materials" in root.child(i).text(0):
                mat_root = root.child(i)
                break
        assert mat_root is not None
        assert mat_root.childCount() == len(real_model.materials)

    def test_repopulate_clears_tree(self, panel, real_model):
        panel.populate(real_model)
        n1 = panel._tree.invisibleRootItem().childCount()
        panel.populate(real_model)
        n2 = panel._tree.invisibleRootItem().childCount()
        assert n1 == n2   # no duplicate top-level items

    def test_group_items_show_element_count(self, panel_loaded, real_model):
        # All group items should mention their element count
        for item in panel_loaded._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "group":
                name = data[1]
                n = len(real_model.groups[name])
                assert str(n) in item.text(0)


# ── Group visibility ──────────────────────────────────────────────────────────

class TestGroupVisibility:
    def test_uncheck_emits_signal(self, panel_loaded, real_model, qapp):
        first_group = sorted(real_model.groups.keys())[0]
        received = []
        panel_loaded.group_visibility_changed.connect(
            lambda name, vis: received.append((name, vis))
        )
        # Find the item and uncheck it
        for item in panel_loaded._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "group" and data[1] == first_group:
                item.setCheckState(0, Qt.CheckState.Unchecked)
                break
        assert len(received) == 1
        assert received[0] == (first_group, False)

    def test_uncheck_calls_scene(self, panel, real_model):
        mock_scene = MagicMock()
        panel.set_scene(mock_scene)
        panel.populate(real_model)

        first_group = sorted(real_model.groups.keys())[0]
        for item in panel._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "group" and data[1] == first_group:
                item.setCheckState(0, Qt.CheckState.Unchecked)
                break
        mock_scene.set_group_visibility.assert_called_with(first_group, False)

    def test_group_selected_signal(self, panel_loaded, real_model):
        first_group = sorted(real_model.groups.keys())[0]
        received = []
        panel_loaded.group_selected.connect(received.append)
        # Find item and simulate click
        for item in panel_loaded._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "group" and data[1] == first_group:
                panel_loaded._on_item_clicked(item, 0)
                break
        assert received == [first_group]


# ── Show / Hide all ───────────────────────────────────────────────────────────

class TestShowHideAll:
    def test_hide_all_groups(self, panel, real_model):
        mock_scene = MagicMock()
        panel.set_scene(mock_scene)
        panel.populate(real_model)

        panel.hide_all_groups()

        assert panel.checked_groups() == []
        mock_scene.set_all_groups_visibility.assert_called_with(False)

    def test_show_all_groups(self, panel, real_model):
        mock_scene = MagicMock()
        panel.set_scene(mock_scene)
        panel.populate(real_model)

        panel.hide_all_groups()
        panel.show_all_groups()

        assert len(panel.checked_groups()) == len(real_model.groups)
        mock_scene.set_all_groups_visibility.assert_called_with(True)

    def test_show_hide_does_not_fire_individual_signals(self, panel, real_model):
        """Batch show/hide must not emit group_visibility_changed per group."""
        panel.populate(real_model)
        received = []
        panel.group_visibility_changed.connect(received.append)
        panel.hide_all_groups()
        assert received == []   # _populating guard suppresses individual signals

    def test_no_scene_show_all_noop(self, panel, real_model):
        """show_all_groups without set_scene must not crash."""
        panel.populate(real_model)
        panel.show_all_groups()   # no scene attached — should be silent


# ── SceneManager integration (real SceneManager + visibility filter) ──────────

class TestSceneIntegration:
    def test_hide_group_reduces_cell_count(self, panel, real_model):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        panel.set_scene(scene)
        panel.populate(real_model)

        full_cells = scene._solid_mesh.n_cells

        # Hide first group
        first_group = sorted(real_model.groups.keys())[0]
        for item in panel._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "group" and data[1] == first_group:
                item.setCheckState(0, Qt.CheckState.Unchecked)
                break

        assert scene._solid_mesh.n_cells < full_cells
        scene.close()

    def test_show_all_restores_full_mesh(self, panel, real_model):
        from fahts.renderer.scene_manager import SceneManager
        scene = SceneManager(off_screen=True)
        scene.load_model(real_model)
        panel.set_scene(scene)
        panel.populate(real_model)

        full_cells = scene._solid_mesh.n_cells

        panel.hide_all_groups()
        panel.show_all_groups()

        assert scene._solid_mesh.n_cells == full_cells
        scene.close()
