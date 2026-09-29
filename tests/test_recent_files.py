"""Tests for Phase 1.9 — recent-files feature in MainWindow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyvistaqt")

from PyQt6.QtWidgets import QApplication  # noqa: E402
from fahts.gui.main_window import MainWindow  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(qapp):
    """Fresh MainWindow; recent files cleared to avoid QSettings contamination."""
    w = MainWindow()
    w._recent_files.clear()
    w._rebuild_recent_menu()
    yield w
    w._plotter.close()
    w.close()


@pytest.fixture(scope="module")
def fem_path():
    p = Path(__file__).parents[1] / "examples" / "models" / "model_file.fem"
    if not p.exists():
        pytest.skip("model_file.fem not found")
    return p


# ── Menu structure ────────────────────────────────────────────────────────────

class TestMenuStructure:
    def test_recent_menu_exists(self, window):
        assert window._recent_menu is not None

    def test_recent_menu_title(self, window):
        assert "Recent" in window._recent_menu.title()

    def test_recent_menu_in_file_menu(self, window):
        mb = window.menuBar()
        file_menu = None
        for action in mb.actions():
            if "File" in action.text():
                file_menu = action.menu()
                break
        assert file_menu is not None
        submenu_titles = [a.menu().title() if a.menu() else "" for a in file_menu.actions()]
        assert any("Recent" in t for t in submenu_titles)

    def test_recent_menu_shows_placeholder_when_empty(self, window):
        actions = window._recent_menu.actions()
        assert len(actions) == 1
        assert not actions[0].isEnabled()

    def test_recent_menu_no_clear_when_empty(self, window):
        labels = [a.text() for a in window._recent_menu.actions()]
        assert not any("Clear" in lbl for lbl in labels)


# ── _add_to_recent ────────────────────────────────────────────────────────────

class TestAddToRecent:
    def test_add_one_file(self, window, fem_path):
        window._add_to_recent(fem_path)
        assert len(window._recent_files) == 1
        assert window._recent_files[0].name == fem_path.name

    def test_add_prepends(self, window, tmp_path):
        f1 = tmp_path / "a.fem"
        f2 = tmp_path / "b.fem"
        f1.touch()
        f2.touch()
        window._add_to_recent(f1)
        window._add_to_recent(f2)
        assert window._recent_files[0].name == "b.fem"
        assert window._recent_files[1].name == "a.fem"

    def test_deduplication_moves_to_top(self, window, tmp_path):
        f1 = tmp_path / "a.fem"
        f2 = tmp_path / "b.fem"
        f1.touch()
        f2.touch()
        window._add_to_recent(f1)
        window._add_to_recent(f2)
        window._add_to_recent(f1)
        assert window._recent_files[0].name == "a.fem"
        assert len(window._recent_files) == 2   # no duplicates

    def test_max_limit_enforced(self, window, tmp_path):
        for i in range(window._MAX_RECENT + 3):
            f = tmp_path / f"model_{i}.fem"
            f.touch()
            window._add_to_recent(f)
        assert len(window._recent_files) == window._MAX_RECENT

    def test_uses_resolved_path(self, window, tmp_path):
        f = tmp_path / "model.fem"
        f.touch()
        window._add_to_recent(f)
        window._add_to_recent(f.resolve())   # same file, different spelling
        assert len(window._recent_files) == 1


# ── _rebuild_recent_menu ──────────────────────────────────────────────────────

class TestRebuildRecentMenu:
    def test_menu_has_entry_after_add(self, window, tmp_path):
        f = tmp_path / "test.fem"
        f.touch()
        window._add_to_recent(f)
        labels = [a.text() for a in window._recent_menu.actions()]
        assert any("test.fem" in lbl for lbl in labels)

    def test_menu_entry_count_matches_files(self, window, tmp_path):
        for i in range(3):
            f = tmp_path / f"m{i}.fem"
            f.touch()
            window._add_to_recent(f)
        # 3 file entries + separator + Clear = 5 items
        assert window._recent_menu.actions()[0].isEnabled()
        file_actions = [
            a for a in window._recent_menu.actions()
            if a.isEnabled() and not a.isSeparator() and "Clear" not in a.text()
        ]
        assert len(file_actions) == 3

    def test_menu_has_clear_action_when_files_present(self, window, tmp_path):
        f = tmp_path / "x.fem"
        f.touch()
        window._add_to_recent(f)
        labels = [a.text() for a in window._recent_menu.actions()]
        assert any("Clear" in lbl for lbl in labels)

    def test_nonexistent_file_greyed_out(self, window, tmp_path):
        f = tmp_path / "ghost.fem"
        f.touch()
        window._add_to_recent(f)
        f.unlink()   # remove after adding
        window._rebuild_recent_menu()   # force refresh
        file_action = next(
            a for a in window._recent_menu.actions()
            if "ghost.fem" in a.text()
        )
        assert not file_action.isEnabled()

    def test_existing_file_is_enabled(self, window, tmp_path):
        f = tmp_path / "real.fem"
        f.touch()
        window._add_to_recent(f)
        file_action = next(
            a for a in window._recent_menu.actions()
            if "real.fem" in a.text()
        )
        assert file_action.isEnabled()

    def test_mnemonic_numbers_applied(self, window, tmp_path):
        for i in range(3):
            f = tmp_path / f"n{i}.fem"
            f.touch()
            window._add_to_recent(f)
        labels = [a.text() for a in window._recent_menu.actions() if ".fem" in a.text()]
        assert any("1." in lbl or "&1." in lbl for lbl in labels)


# ── _on_clear_recent ──────────────────────────────────────────────────────────

class TestClearRecent:
    def test_clear_empties_list(self, window, tmp_path):
        for i in range(3):
            f = tmp_path / f"c{i}.fem"
            f.touch()
            window._add_to_recent(f)
        window._on_clear_recent()
        assert window._recent_files == []

    def test_clear_restores_placeholder(self, window, tmp_path):
        f = tmp_path / "d.fem"
        f.touch()
        window._add_to_recent(f)
        window._on_clear_recent()
        actions = window._recent_menu.actions()
        assert len(actions) == 1
        assert not actions[0].isEnabled()

    def test_clear_button_clears_via_menu(self, window, tmp_path):
        f = tmp_path / "e.fem"
        f.touch()
        window._add_to_recent(f)
        clear_action = next(
            a for a in window._recent_menu.actions() if "Clear" in a.text()
        )
        clear_action.trigger()
        assert window._recent_files == []


# ── Integration with open_file ────────────────────────────────────────────────

class TestOpenFileIntegration:
    def test_open_file_adds_to_recent(self, window, fem_path):
        window.open_file(fem_path)
        assert len(window._recent_files) >= 1
        assert window._recent_files[0].name == fem_path.name

    def test_open_file_twice_no_duplicate(self, window, fem_path):
        window.open_file(fem_path)
        window.open_file(fem_path)
        assert len(window._recent_files) == 1

    def test_failed_open_does_not_add_to_recent(self, window, monkeypatch):
        monkeypatch.setattr("fahts.gui.main_window.QMessageBox.critical", lambda *a, **k: None)
        window.open_file(Path("/nonexistent/model.fem"))
        assert window._recent_files == []

    def test_recent_menu_populated_after_open(self, window, fem_path):
        window.open_file(fem_path)
        labels = [a.text() for a in window._recent_menu.actions()]
        assert any(fem_path.name in lbl for lbl in labels)
