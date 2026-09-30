"""Local (jet) peak fire zone on the vessel model's 3-D wall (known issue #10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.coupling import VesselFireModel, VesselFireOptions
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[2] / "regression" / "process" / "cases"


def _model(case_id, steel, **opts):
    return VesselFireModel(read_case(CASES / case_id), VesselFireOptions(t_end=1.0, **opts), steel)


def test_uniform_fire_has_no_peak_regions(steel):
    m = _model("M06-0003", steel)
    assert m.peak is None and m.frac["peak_dry"] == m.frac["peak_wet"] == 0.0
    assert not m.wall3d.mesh.outer_area_peak.any()


@pytest.mark.parametrize("case_id, peak_wet_share", [("M06-0030", None),   # gas only
                                                      ("M06-0069", 1.0),    # bottom, wetted
                                                      ("M06-0070", 0.0)])   # top, dry
def test_peak_regions_follow_the_zone_and_the_liquid(steel, case_id, peak_wet_share):
    m = _model(case_id, steel)
    assert sum(m.frac.values()) == pytest.approx(1.0)
    f_pk = m.frac["peak_dry"] + m.frac["peak_wet"]
    assert f_pk == pytest.approx(0.2 * 90 / 360, rel=1e-9)        # xi 0.4-0.6, 90 deg
    if peak_wet_share is not None:
        # the wetted arc is resolved by the mesh faces (face centres): exact for these zones
        assert m.frac["peak_wet"] / f_pk == pytest.approx(peak_wet_share, abs=1e-9)


def test_peak_zone_can_be_switched_off(steel):
    m = _model("M06-0070", steel, peak_zone=False)
    assert m.peak is None and m.frac["peak_dry"] + m.frac["peak_wet"] == 0.0


def test_peak_zone_sees_the_peak_flux(steel):
    m = _model("M06-0030", steel)
    w = m.wall3d
    T = np.full(1, 293.15)
    one = np.ones(1)
    q_bg = w._outer_rates(w.bc_background, one, T, 10.0)[0][0]
    q_pk = w._outer_rates(w.bc_peak, one, T, 10.0)[0][0]
    assert q_pk > 2.0 * q_bg                                     # 250 vs 100 kW/m2 incident
