"""Fire section: heat-load table, quick fill, peak zone, sketches, validation of bad cells."""

from __future__ import annotations

from pathlib import Path

import pytest

from fahts.coupling.vessel_case import VesselCase
from fahts.gui.process.contents_form import ContentsForm
from fahts.gui.process.fire_form import FireForm
from fahts.gui.process.workspace import ProcessWorkspace

CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"


def test_jet_fire_deck_round_trips(qapp):
    ws = ProcessWorkspace()
    ws.import_deck(CASES / "M06-0070")                 # LPG, jet on top
    c = ws.current_case()
    assert c.heat_load.peak.attack_deg == 0.0 and c.heat_load.peak.circ_deg == 90.0
    assert max(c.heat_load.q_peak_kW_m2) == 250.0 and max(c.heat_load.q_background_kW_m2) == 100.0
    f = ws.form(FireForm)
    assert "5.00 % of the shell" in f.zone_info.text()      # 0.2 of length x 90/360


def test_quick_fill_and_rows(qapp):
    f = FireForm()
    f.set_case(VesselCase())
    f.q_bg.setValue(50.0)
    f.q_pk.setValue(200.0)
    f.q_until.setValue(1800.0)
    f.btn_fill.click()
    assert f.rows() == [(0.0, 50.0, 200.0), (1800.0, 50.0, 200.0), (1801.0, 0.0, 0.0)]
    f.btn_add.click()
    assert f.rows()[-1] == (2401.0, 0.0, 0.0)
    case = VesselCase()
    f.apply(case)
    assert case.heat_load.times_s[-1] == 2401.0 and case.validate() == []


def test_uniform_fire_note(qapp):
    f = FireForm()
    f.set_case(VesselCase())                          # default: 100 / 100 kW/m2
    assert "no effect" in f.zone_info.text()


def test_bad_cell_is_caught(qapp):
    f = FireForm()
    f.set_case(VesselCase())
    f.table.item(1, 1).setText("abc")
    case = VesselCase()
    f.apply(case)
    assert any("must be a number" in e for e in case.validate())


def test_sketch_follows_liquid_level(qapp):
    ws = ProcessWorkspace()
    ws.form(ContentsForm).hc.setValue(1.2)
    assert ws.form(FireForm)._level == pytest.approx(1.2)
