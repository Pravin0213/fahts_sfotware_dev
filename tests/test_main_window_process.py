"""Main window: Process menu and opening vessel cases (needs a display, like other GUI tests)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyvistaqt")

from PyQt6.QtWidgets import QApplication  # noqa: E402
from fahts.gui.main_window import MainWindow  # noqa: E402

DECK = Path(__file__).resolve().parent / "regression" / "process" / "cases" / "M09-0001"


@pytest.fixture(scope="session")
def qapp():
    import sys
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w._process_ws.load_case(w._process_ws.current_case())    # clean: no unsaved prompt
    w.close()


def test_process_menu_and_tabs(window):
    menus = [a.text() for a in window.menuBar().actions()]
    assert "&Process" in menus
    assert window._workspace_tabs.count() == 2


def test_open_deck_and_case_file(window, tmp_path):
    window.open_process_case(DECK)
    assert window._workspace_tabs.currentWidget() is window._process_ws
    c = window._process_ws.current_case()
    assert c.blowdown.enabled and c.psv.enabled
    window._process_ws.save(tmp_path / "c.vcase.json")
    window.open_process_case(tmp_path / "c.vcase.json")
    assert window._process_ws.path.name == "c.vcase.json"
