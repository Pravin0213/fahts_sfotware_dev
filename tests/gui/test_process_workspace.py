"""Process workspace: forms round-trip a VesselCase; save / open / import; dirty state."""

from __future__ import annotations

from pathlib import Path

import pytest

from fahts.coupling.vessel_case import BlowdownLineSpec, VesselCase
from fahts.gui.process.forms import OptionsForm, ReliefForm, VesselForm
from fahts.gui.process.workspace import ProcessWorkspace

CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"


def _approx_equal(a: dict, b: dict, path=""):
    assert type(a) is type(b) or {type(a), type(b)} <= {int, float}, path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _approx_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _approx_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def _rich_case() -> VesselCase:
    c = VesselCase(name="rich", description="test")
    c.vessel.wall_m, c.vessel.tag = 0.0874, "V-7"
    c.blowdown.enabled, c.blowdown.orifice_mm, c.blowdown.delay_s = True, 25.4, 30.0
    c.blowdown.line = BlowdownLineSpec(168.3, 7.11, 35.0)
    c.psv.enabled, c.psv.characteristic = True, "square"
    c.ambient.convection, c.ambient.h_W_m2K = "fixed", 12.5
    c.stress.basis, c.stress.factor = "yield", 0.85
    c.run.t_end_s, c.run.dt_s = 1800.0, 0.5
    c.options = {"h_corr": "churchill_chu", "peak_zone": False, "wall_cells": 20}
    return c


def test_forms_round_trip_the_case(qapp):
    ws = ProcessWorkspace()
    case = _rich_case()
    ws.load_case(case)
    _approx_equal(ws.current_case().to_dict(), case.to_dict())
    assert not ws.is_dirty


def test_editing_marks_dirty_and_updates_case(qapp):
    ws = ProcessWorkspace()
    titles = []
    ws.title_changed.connect(titles.append)
    ws.form(VesselForm).wall.setValue(75.0)
    assert ws.is_dirty and titles[-1].endswith("*")
    assert ws.current_case().vessel.wall_m == pytest.approx(0.075)
    ws.form(ReliefForm).bdv_box.setChecked(True)
    assert ws.current_case().blowdown.enabled


def test_options_store_only_changes_from_default(qapp):
    ws = ProcessWorkspace()
    assert ws.current_case().options == {}
    f = ws.form(OptionsForm)
    w = f.widgets["h_corr"]
    w.setCurrentIndex(w.property("choices").index("churchill_chu"))
    assert ws.current_case().options == {"h_corr": "churchill_chu"}


def test_save_open_and_import(qapp, tmp_path):
    ws = ProcessWorkspace()
    ws.load_case(_rich_case())
    ws.save(tmp_path / "x.vcase.json")
    assert not ws.is_dirty and ws.path.name == "x.vcase.json"
    ws2 = ProcessWorkspace()
    ws2.open_file(tmp_path / "x.vcase.json")
    _approx_equal(ws2.current_case().to_dict(), ws.current_case().to_dict())
    warnings = ws2.import_deck(CASES / "M09-0001")
    assert warnings and ws2.is_dirty
    c = ws2.current_case()
    assert c.blowdown.enabled and c.psv.enabled and c.validate() == []


def test_validation_messages(qapp):
    ws = ProcessWorkspace()
    ws.form(VesselForm).D.setValue(0.5)                    # smaller than default liquid? no liquid
    assert ws.validate() == []
    ws.form(ReliefForm).psv_box.setChecked(True)
    ws.form(ReliefForm).psv_reseat.setValue(500.0)
    errs = ws.validate()
    assert any("reseat" in e for e in errs)
    assert ws.messages.count() == len(errs)


def test_spin_boxes_keep_exact_values_until_edited(qapp):
    from fahts.gui.process.fields import dspin
    w = dspin(0.0, 10.0, 2)
    w.setValue(1.01325)
    assert w.value() == 1.01325                       # not display-rounded
    w.lineEdit().setText(w.locale().toString(2.5, "f", 2))   # typed in the user's locale
    w.interpretText()
    assert w.value() == 2.5                           # user edit wins
    w.setValue(1.01325)
    w.stepBy(1)
    assert w.value() == pytest.approx(1.11)           # arrow / wheel edit wins too


def test_3d_preview_follows_edits_without_a_display(qapp):
    """The 3-D geometry is rebuilt on edits; the VTK widget itself is not created offscreen."""
    from fahts.gui.process.contents_form import ContentsForm
    ws = ProcessWorkspace()
    assert ws.vessel_view.plotter is None
    ws.form(ContentsForm).hc.setValue(1.0)
    ws._refresh_preview()                           # the debounce timer would do this
    assert ws.vessel_view._geom.level == pytest.approx(1.0)
    assert ws.vessel_view._geom.region_fractions()["wet"] == pytest.approx(0.5, abs=0.01)
