"""3-D vessel view widget (needs a display, like the other VTK tests)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
    assert {"wall", "zone", "head0", "head1", "liquid"} <= _actors(view)
    view.show_liquid.setChecked(False)
    assert "liquid" not in _actors(view)


def test_result_painting_and_time_slider(view):
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")
    case.run.t_end_s = 120.0
    res = run_case(case)
    view.show_result(res)
    assert view.slider.isEnabled() and view.slider.maximum() == len(res.series) - 1
    wall = view._geom.wall
    ts, last = res.series, len(res.series) - 1
    T = wall.cell_data["T_C"]                                      # through thickness (default)
    assert np.nanmax(T) <= ts.peak_T_out_C.iloc[last] + 1e-9       # hottest: jet, outer surface
    assert np.nanmax(T) > ts.peak_T_mean_C.iloc[last]              # gradient, not the mean
    view.mode.setCurrentIndex(2)                                   # through-wall mean
    assert np.nanmax(wall.cell_data["T_C"]) == pytest.approx(ts.peak_T_mean_C.iloc[last])
    view.slider.setValue(0)
    assert np.nanmax(view._geom.wall.cell_data["T_C"]) == pytest.approx(ts.peak_T_mean_C.iloc[0])


def test_cutaway_and_thickness_scale(view):
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")
    view.show_case(case)
    full = view._geom.wall_surface("none").n_cells
    view.cutaway.setCurrentIndex(2)                                # half
    assert "head0" not in _actors(view) and "wall" in _actors(view)
    view.scale.setCurrentIndex(2)                                  # x5
    assert view._geom.R_out == pytest.approx(view._geom.R + 5 * case.vessel.wall_m)
    assert "×5" in view.legend.text() and full > 0
