"""Run a case from the workspace (background thread), show results, export, cancel."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from fahts.gui.process.results_view import summary_lines
from fahts.gui.process.workspace import ProcessWorkspace

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"


def _wait(ws, timeout=120.0):
    t0 = time.time()
    while ws.is_running and time.time() - t0 < timeout:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    assert not ws.is_running, "run did not finish"


def test_run_shows_results_and_exports(qapp, tmp_path):
    ws = ProcessWorkspace()
    ws.import_deck(CASES / "M06-0030")                    # CH4, jet at the bottom
    case = ws.current_case()
    case.run.t_end_s = 300.0
    ws.load_case(case)
    got = []
    ws.run_finished.connect(got.append)
    assert ws.start_run() is not None and ws.is_running
    _wait(ws)
    assert got and ws.results.isVisibleTo(ws) and not ws.results_placeholder.isVisibleTo(ws)
    res = got[0]
    assert res.series.Time.iloc[-1] == 300.0 and "peak_T_mean_C" in res.series
    labels = dict(summary_lines(res))
    assert "bara" in labels["Peak pressure"] and labels["Case"] == case.name
    assert ws.results.summary.rowCount() == len(labels)
    assert ws.results.fail_table.rowCount() == len(res.failures)
    ws.results.export_csv(tmp_path / "r.csv")
    ws.results.export_xlsx(tmp_path / "r.xlsx")
    assert len(pd.read_csv(tmp_path / "r.csv")) == len(res.series)
    assert set(pd.ExcelFile(tmp_path / "r.xlsx").sheet_names) == {
        "summary", "time series", "failure times", "stress"}


def test_invalid_case_does_not_start(qapp):
    ws = ProcessWorkspace()
    case = ws.current_case()
    case.vessel.wall_m = 0.0
    ws.load_case(case)
    assert ws.start_run() is None and not ws.is_running


def test_stop_cancels_the_run(qapp):
    ws = ProcessWorkspace()
    msgs = []
    ws.status_message.connect(msgs.append)
    ws.start_run()                                          # default: 3600 s, CH4 fire
    ws.stop_run()
    _wait(ws)
    assert msgs[-1] == "Run cancelled" and ws.btn_run.isEnabled()


def test_regions_without_area_are_not_shown(qapp):
    """Gas-only vessel with a jet on top: no wetted regions, only dry background + peak."""
    from fahts.coupling import run_case
    from fahts.coupling.vessfire_import import case_from_vessfire_deck
    from fahts.gui.process.results_view import regions_with_area
    case, _ = case_from_vessfire_deck(CASES / "M06-0030")       # CH4, no liquid, peak bottom
    case.run.t_end_s = 60.0
    res = run_case(case)
    assert regions_with_area(res) == ["background", "peak"]
    labels = dict(summary_lines(res))
    assert labels["Wall regions"] == "dry wall, peak zone (dry)"
