"""3-D vessel view widget (needs a display, like the other VTK tests)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyvistaqt")
pytest.importorskip("CoolProp")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from fahts.coupling import run_case  # noqa: E402
from fahts.coupling.vessfire_import import case_from_vessfire_deck  # noqa: E402
from fahts.gui.process.vessel_view import VesselView  # noqa: E402

CASES = Path(__file__).resolve().parent / "regression" / "process" / "cases"


@pytest.fixture(scope="session")
def qapp():
    import sys
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def view(qapp):
    v = VesselView()
    v.ensure_plotter()
    yield v
    v.plotter.close()


def _actors(v):
    return set(v.plotter.renderer.actors)


def test_case_preview_draws_shell_zone_heads_liquid(view):
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")        # LPG, jet on top
    view.show_case(case)
    assert {"shell", "zone", "head0", "head1", "liquid"} <= _actors(view)
    view.show_liquid.setChecked(False)
    assert "liquid" not in _actors(view)


def test_result_painting_and_time_slider(view):
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")
    case.run.t_end_s = 120.0
    res = run_case(case)
    view.show_result(res)
    assert view.slider.isEnabled() and view.slider.maximum() == len(res.series) - 1
    shell = view._geom.shell
    assert "T_C" in shell.cell_data and shell.cell_data["T_C"].max() > 20.0
    view.slider.setValue(0)
    assert shell.cell_data["T_C"].max() == pytest.approx(res.series.background_T_mean_C.iloc[0])
    view.mode.setCurrentIndex(2)                                  # outer surface
    assert shell.cell_data["T_C"].max() >= res.series.peak_T_out_C.iloc[0] - 1e-9
