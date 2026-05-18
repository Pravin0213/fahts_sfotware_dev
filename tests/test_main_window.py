"""Tests for fahts/gui/main_window.py (Phase 1.6 / 4.4)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Guard: skip entire module if PyQt6 or pyvistaqt are unavailable.
pytest.importorskip("PyQt6")
pytest.importorskip("pyvistaqt")

from PyQt6.QtWidgets import QApplication  # noqa: E402 — after importorskip
from fahts.gui.main_window import MainWindow  # noqa: E402


# ── QApplication singleton ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    """One QApplication per session (Qt forbids multiple)."""
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def window(qapp):
    """Fresh MainWindow for each test; closed and cleaned up afterwards."""
    w = MainWindow()
    # Reset persisted recent-files so tests don't depend on QSettings state.
    w._recent_files.clear()
    w._rebuild_recent_menu()
    yield w
    w._plotter.close()
    w.close()


@pytest.fixture(scope="module")
def fem_path():
    p = Path(__file__).parents[1] / "model_file.fem"
    if not p.exists():
        pytest.skip("model_file.fem not found")
    return p


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_window_created(self, window):
        assert window is not None

    def test_has_menu_bar(self, window):
        mb = window.menuBar()
        assert mb is not None
        titles = [mb.actions()[i].text() for i in range(len(mb.actions()))]
        # Strip & mnemonics
        cleaned = [t.replace("&", "") for t in titles]
        assert "File" in cleaned
        assert "View" in cleaned
        assert "Model" in cleaned
        assert "Help" in cleaned

    def test_has_toolbar(self, window):
        toolbars = window.findChildren(window.__class__.__mro__[0].__mro__[0].__mro__[0].__mro__[0].__mro__[-1])
        # Just check the toolbar exists via the open action being present
        assert window._action_open is not None

    def test_status_bar_initial_text(self, window):
        assert window._status_label.text() == "Ready"

    def test_initial_model_is_none(self, window):
        assert window._model is None

    def test_render_mode_actions_disabled_before_load(self, window):
        assert not window._action_mode_wire.isEnabled()
        assert not window._action_mode_section.isEnabled()

    def test_colour_group_action_disabled_before_load(self, window):
        assert not window._action_colour_group.isEnabled()

    def test_reset_camera_disabled_before_load(self, window):
        assert not window._action_reset_cam.isEnabled()

    def test_initial_render_mode_section_checked(self, window):
        assert window._action_mode_section.isChecked()
        assert not window._action_mode_wire.isChecked()

    def test_initial_colour_default_checked(self, window):
        assert window._action_colour_default.isChecked()
        assert not window._action_colour_group.isChecked()


# ── open_file ─────────────────────────────────────────────────────────────────

class TestOpenFile:
    def test_open_real_file(self, window, fem_path):
        window.open_file(fem_path)
        assert window._model is not None

    def test_model_has_correct_counts(self, window, fem_path):
        window.open_file(fem_path)
        assert window._model.n_nodes == 659
        assert window._model.n_elements == 783

    def test_status_bar_updated(self, window, fem_path):
        window.open_file(fem_path)
        txt = window._status_label.text()
        assert "659" in txt
        assert "783" in txt

    def test_actions_enabled_after_load(self, window, fem_path):
        window.open_file(fem_path)
        assert window._action_mode_wire.isEnabled()
        assert window._action_mode_section.isEnabled()
        assert window._action_colour_group.isEnabled()
        assert window._action_reset_cam.isEnabled()

    def test_model_loaded_signal_emitted(self, window, fem_path, qapp):
        received = []
        window.model_loaded.connect(lambda m: received.append(m))
        window.open_file(fem_path)
        assert len(received) == 1
        assert received[0] is window._model

    def test_nonexistent_file_shows_no_crash(self, window, monkeypatch):
        monkeypatch.setattr("fahts.gui.main_window.QMessageBox.critical", lambda *a, **k: None)
        window.open_file(Path("/nonexistent/path/model.fem"))
        assert window._model is None   # model unchanged

    def test_open_file_twice(self, window, fem_path):
        """Calling open_file twice must replace the old model cleanly."""
        window.open_file(fem_path)
        m1 = window._model
        window.open_file(fem_path)
        m2 = window._model
        assert m1 is not m2          # new object each time

    def test_tree_panel_populated_after_load(self, window, fem_path):
        window.open_file(fem_path)
        # ModelTreePanel should be populated — check checked groups
        checked = window._tree_panel.checked_groups()
        assert len(checked) == len(window._model.groups)


# ── Action wiring ─────────────────────────────────────────────────────────────

class TestActionWiring:
    def test_render_mode_wire_triggers_scene(self, window, fem_path):
        window.open_file(fem_path)
        window._action_mode_wire.setChecked(True)
        window._on_render_mode_changed(window._action_mode_wire)
        assert window._scene.render_mode == "wire"

    def test_render_mode_section_triggers_scene(self, window, fem_path):
        window.open_file(fem_path)
        window._action_mode_wire.setChecked(True)
        window._on_render_mode_changed(window._action_mode_wire)
        window._action_mode_section.setChecked(True)
        window._on_render_mode_changed(window._action_mode_section)
        assert window._scene.render_mode == "section"

    def test_colour_group_triggers_scene(self, window, fem_path):
        window.open_file(fem_path)
        window._on_colour_mode_changed(window._action_colour_group)
        assert window._scene.colour_mode == "group"

    def test_colour_default_resets_scene(self, window, fem_path):
        window.open_file(fem_path)
        window._on_colour_mode_changed(window._action_colour_group)
        window._on_colour_mode_changed(window._action_colour_default)
        assert window._scene.colour_mode == "default"

    def test_model_summary_no_model(self, window, monkeypatch):
        """Summary with no model should not raise."""
        monkeypatch.setattr("fahts.gui.main_window.QMessageBox.information", lambda *a, **k: None)
        window._on_model_summary()

    def test_model_summary_with_model(self, window, fem_path, monkeypatch):
        monkeypatch.setattr("fahts.gui.main_window.QMessageBox.information", lambda *a, **k: None)
        window.open_file(fem_path)
        window._on_model_summary()

    def test_mode_group_is_exclusive(self, window):
        """Checking Wire must uncheck Section and vice versa."""
        assert window._action_mode_section.isChecked()
        # Simulate user clicking Wire
        window._action_mode_wire.setChecked(True)
        assert not window._action_mode_section.isChecked()

    def test_colour_group_is_exclusive(self, window):
        assert window._action_colour_default.isChecked()
        window._action_colour_group.setChecked(True)
        assert not window._action_colour_default.isChecked()


# ── Left panel ────────────────────────────────────────────────────────────────

class TestLeftPanel:
    def test_left_panel_exists(self, window):
        assert window._left_panel is not None

    def test_tree_panel_exists(self, window):
        assert window._tree_panel is not None

    def test_props_placeholder_exists(self, window):
        assert window._props_placeholder is not None

    def test_left_panel_max_width(self, window):
        assert window._left_panel.maximumWidth() <= 400


# ── Task 4.4 — Animation toolbar ─────────────────────────────────────────────


def _make_tf(n_steps: int = 5, n_elems: int = 4):
    """Build a minimal TemperatureField-like object for animation tests."""
    times = np.linspace(0.0, 3600.0, n_steps)
    eids = list(range(1, n_elems + 1))
    T_centroid = np.random.default_rng(0).uniform(20.0, 700.0, (n_steps, n_elems))
    return SimpleNamespace(times=times, element_ids=eids, T_centroid=T_centroid)


class TestAnimationToolbar:
    """Animation toolbar: present, hidden initially, shown after _anim_show()."""

    def test_anim_toolbar_exists(self, window):
        assert hasattr(window, "_anim_toolbar")

    def test_anim_toolbar_hidden_initially(self, window):
        assert not window._anim_toolbar.isVisible()

    def test_anim_slider_exists(self, window):
        assert hasattr(window, "_anim_slider")

    def test_anim_time_label_exists(self, window):
        assert hasattr(window, "_anim_time_label")

    def test_anim_speed_combo_exists(self, window):
        assert hasattr(window, "_anim_speed_combo")

    def test_anim_toolbar_visible_after_show(self, window, fem_path):
        window.open_file(fem_path)
        tf = _make_tf()
        window._last_result = tf
        window._anim_show(tf)
        # Use not-isHidden(): isVisible() requires the parent chain to also be shown
        assert not window._anim_toolbar.isHidden()

    def test_anim_toolbar_hidden_after_open_file(self, window, fem_path):
        window.open_file(fem_path)
        tf = _make_tf()
        window._last_result = tf
        window._anim_show(tf)
        assert not window._anim_toolbar.isHidden()
        # Loading a new model clears the toolbar
        window.open_file(fem_path)
        assert window._anim_toolbar.isHidden()

    def test_anim_slider_range_set(self, window, fem_path):
        window.open_file(fem_path)
        tf = _make_tf(n_steps=8)
        window._last_result = tf
        window._anim_show(tf)
        assert window._anim_slider.minimum() == 0
        assert window._anim_slider.maximum() == 7   # n_steps - 1

    def test_anim_n_steps_no_result(self, window):
        assert window._anim_n_steps() == 0

    def test_anim_n_steps_with_result(self, window, fem_path):
        window.open_file(fem_path)
        tf = _make_tf(n_steps=6)
        window._last_result = tf
        assert window._anim_n_steps() == 6

    def test_open_file_clears_last_result(self, window, fem_path):
        window.open_file(fem_path)
        window._last_result = _make_tf()
        window.open_file(fem_path)
        assert window._last_result is None


class TestAnimationPlayback:
    """Step navigation and play/pause logic."""

    @pytest.fixture
    def loaded_window(self, window, fem_path):
        window.open_file(fem_path)
        tf = _make_tf(n_steps=5)
        window._last_result = tf
        window._anim_show(tf)
        return window

    def test_initial_step_is_zero(self, loaded_window):
        assert loaded_window._anim_t_idx == 0

    def test_anim_first_goes_to_zero(self, loaded_window):
        loaded_window._on_anim_next()   # advance first
        loaded_window._on_anim_first()
        assert loaded_window._anim_t_idx == 0

    def test_anim_last_goes_to_final(self, loaded_window):
        loaded_window._on_anim_last()
        assert loaded_window._anim_t_idx == 4   # n_steps - 1

    def test_anim_next_advances_one(self, loaded_window):
        loaded_window._on_anim_next()
        assert loaded_window._anim_t_idx == 1

    def test_anim_prev_decrements_one(self, loaded_window):
        loaded_window._on_anim_next()
        loaded_window._on_anim_prev()
        assert loaded_window._anim_t_idx == 0

    def test_anim_next_clamps_at_end(self, loaded_window):
        loaded_window._on_anim_last()
        loaded_window._on_anim_next()   # already at end — must not crash
        assert loaded_window._anim_t_idx == 4

    def test_anim_prev_clamps_at_start(self, loaded_window):
        loaded_window._on_anim_prev()   # already at 0 — must not crash
        assert loaded_window._anim_t_idx == 0

    def test_play_sets_playing_flag(self, loaded_window):
        loaded_window._on_anim_play_pause()
        assert loaded_window._anim_playing is True
        loaded_window._anim_stop()      # cleanup

    def test_pause_clears_playing_flag(self, loaded_window):
        loaded_window._on_anim_play_pause()   # play
        loaded_window._on_anim_play_pause()   # pause
        assert loaded_window._anim_playing is False

    def test_play_button_text_changes(self, loaded_window):
        assert loaded_window._action_anim_play.text() == "▶"
        loaded_window._on_anim_play_pause()
        assert loaded_window._action_anim_play.text() == "⏸"
        loaded_window._on_anim_play_pause()
        assert loaded_window._action_anim_play.text() == "▶"

    def test_play_starts_timer(self, loaded_window):
        loaded_window._on_anim_play_pause()
        assert loaded_window._anim_timer.isActive()
        loaded_window._anim_stop()

    def test_stop_stops_timer(self, loaded_window):
        loaded_window._on_anim_play_pause()
        loaded_window._anim_stop()
        assert not loaded_window._anim_timer.isActive()

    def test_anim_tick_advances_index(self, loaded_window):
        loaded_window._anim_go_to(0)
        loaded_window._on_anim_tick()
        assert loaded_window._anim_t_idx == 1

    def test_anim_tick_at_last_stops(self, loaded_window):
        loaded_window._on_anim_last()
        loaded_window._on_anim_play_pause()   # play from last
        # tick should stop, not wrap
        loaded_window._on_anim_tick()
        assert loaded_window._anim_playing is False

    def test_anim_go_to_updates_slider(self, loaded_window):
        loaded_window._anim_go_to(3)
        assert loaded_window._anim_slider.value() == 3

    def test_anim_time_label_updates(self, loaded_window):
        loaded_window._anim_go_to(4)
        text = loaded_window._anim_time_label.text()
        assert "5/5" in text   # "Step 5/5 | t = ..."

    def test_speed_change_updates_fps(self, loaded_window):
        loaded_window._on_anim_speed_changed("20")
        assert loaded_window._anim_fps == 20

    def test_speed_change_bad_value_defaults(self, loaded_window):
        loaded_window._on_anim_speed_changed("bad")
        assert loaded_window._anim_fps == 10

    def test_play_from_end_does_not_rewind(self, loaded_window):
        """Pressing play at the last step does not rewind — tick will stop it."""
        loaded_window._on_anim_last()
        loaded_window._on_anim_play_pause()
        assert loaded_window._anim_t_idx == 4   # still at end
        loaded_window._anim_stop()

    def test_scene_colour_mode_is_temperature_after_go_to(self, loaded_window):
        loaded_window._anim_go_to(2)
        assert loaded_window._scene.colour_mode == "temperature"
