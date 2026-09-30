"""Contents form: composition table incl. pseudo-components, presets, initial-state check."""

from __future__ import annotations

import pytest

from fahts.coupling.vessel_case import VesselCase
from fahts.gui.process.contents_form import ContentsForm
from fahts.gui.process.workspace import ProcessWorkspace

pytest.importorskip("CoolProp")


def test_pseudo_components_round_trip(qapp):
    ws = ProcessWorkspace()
    c = VesselCase()
    c.contents.composition = {"C1": 0.7, "PSEU1": 0.3}
    c.contents.pseudo = {"PSEU1": {"sg": 0.7, "Tb_K": 350.0}}
    c.contents.hc_liquid_depth_m = 1.0
    ws.load_case(c)
    got = ws.current_case().contents
    assert got.composition == {"C1": 0.7, "PSEU1": 0.3}
    assert got.pseudo == {"PSEU1": {"sg": 0.7, "Tb_K": 350.0}}
    assert got.hc_liquid_depth_m == 1.0


def test_table_editing_normalise_and_presets(qapp):
    f = ContentsForm()
    f.set_case(VesselCase())
    f._apply_preset("LPG (propane / n-butane 60/40)")
    case = VesselCase()
    f.apply(case)
    assert case.contents.composition == {"C3": 0.6, "C4": 0.4}
    f._add_row("C2", 0.5)
    assert "Normalise" in f.sum_label.text()
    f._normalise()
    f.apply(case)
    assert sum(case.contents.composition.values()) == pytest.approx(1.0)
    assert case.contents.composition["C2"] == pytest.approx(0.5 / 1.5)
    f.btn_pseudo.click()                       # PSEU1 with default SG / Tb
    f.apply(case)
    assert "PSEU1" in case.contents.pseudo and case.contents.pseudo["PSEU1"]["sg"] == 0.75


def test_blank_fraction_is_caught_by_validation(qapp):
    f = ContentsForm()
    f.set_case(VesselCase())
    f.table.item(0, 1).setText("")
    case = VesselCase()
    f.apply(case)
    assert any("sum to" in e for e in case.validate())


def test_initial_state_check_uses_the_model(qapp):
    ws = ProcessWorkspace()
    c = VesselCase()
    c.contents.composition = {"C3": 0.6, "C4": 0.4}
    c.contents.P0_bara, c.contents.hc_liquid_depth_m = 5.0, 1.0        # below the ~6 bara bubble point
    ws.load_case(c)
    text = ws.form(ContentsForm).check_initial_state(ws.current_case())
    assert "Gas zone" in text and "Liquid zone" in text and "Liquid level 1.000" in text
    c.contents.hc_liquid_depth_m = 3.0                      # fills a 2 m vessel
    ws.load_case(c)
    text = ws.form(ContentsForm).check_initial_state(ws.current_case())
    assert "Fix the inputs" in text and "fill the vessel" in text


def test_initial_state_check_reports_model_errors(qapp):
    """Compressed-liquid LPG (above its bubble point) has no vapour: shown, not raised."""
    c = VesselCase()
    c.contents.composition = {"C3": 0.6, "C4": 0.4}
    c.contents.P0_bara, c.contents.hc_liquid_depth_m = 8.0, 1.0
    f = ContentsForm()
    text = f.check_initial_state(c)
    assert "cannot start" in text and "no vapour phase" in text
