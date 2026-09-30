"""VesselCase: validation, JSON round trip, VessFire deck import, runner (progress/cancel)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fahts.coupling import (SimulationCancelled, VesselCase, VesselFireModel,
                            VesselFireOptions, run_case)
from fahts.coupling.vessel_case import BlowdownLineSpec, MaterialSpec
from fahts.coupling.vessfire_import import case_from_vessfire_deck, read_deck
from fahts.materials import SteelTable, en1993_carbon_steel

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[2] / "regression" / "process" / "cases"


def test_default_case_is_valid_and_round_trips(tmp_path):
    c = VesselCase()
    c.blowdown.enabled = True
    c.blowdown.line = BlowdownLineSpec()
    c.contents.pseudo = {"PSEU1": {"sg": 0.7, "Tb_K": 350.0}}
    c.contents.composition = {"C1": 0.7, "PSEU1": 0.3}
    c.options = {"h_corr": "churchill_chu"}
    assert c.validate() == []
    c.save(tmp_path / "a.vcase.json")
    c2 = VesselCase.load(tmp_path / "a.vcase.json")
    assert c2.to_dict() == c.to_dict()
    assert c2.model_options().h_corr == "churchill_chu"


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.contents.composition.update(C1=0.5), "sum to"),
    (lambda c: setattr(c.contents, "hc_liquid_depth_m", 2.5), "fill the vessel"),
    (lambda c: c.contents.composition.update({"XYZ": 0.0}), "unknown component"),
    (lambda c: (setattr(c.psv, "enabled", True), setattr(c.psv, "reseat_bara", 130.0)),
     "reseat <= set"),
    (lambda c: setattr(c.heat_load, "times_s", [0.0, 0.0]), "increasing"),
    (lambda c: setattr(c.vessel, "wall_m", 0.0), "wall thickness"),
    (lambda c: c.options.update(bogus=1), "unknown model options"),
])
def test_validation_catches_bad_input(mutate, message):
    c = VesselCase()
    mutate(c)
    assert any(message in e for e in c.validate()), c.validate()


def test_schema_guard():
    with pytest.raises(ValueError, match="not a fahts vessel case"):
        VesselCase.from_dict({"schema": "other"})


def test_en1993_material_and_csv(tmp_path):
    m = en1993_carbon_steel()
    at = lambda f, T: float(f(T + 273.15))  # noqa: E731
    assert at(m.cp_at, 20) == pytest.approx(439.8, abs=0.1)
    assert at(m.cp_at, 735) == pytest.approx(5000.0)                  # EN 1993-1-2 eq. (3.2)
    assert at(m.k_at, 900) == pytest.approx(27.3)
    assert at(m.f_yield_at, 600) == pytest.approx(0.47)             # Table 3.1
    assert at(m.f_uts_at, 20) == pytest.approx(1.0)
    assert at(m.f_uts_at, 600) == pytest.approx(0.47 / 1.25)        # Annex A
    m.to_csv(tmp_path / "s.csv")
    m2 = SteelTable.from_csv(tmp_path / "s.csv")
    np.testing.assert_allclose(m2.cp, m.cp, rtol=1e-5)
    c = VesselCase()
    c.vessel.material = MaterialSpec(kind="table", table=m2.to_dict())
    assert c.validate() == [] and c.vessel.material.steel_table().name == m.name


@pytest.mark.parametrize("case_id", sorted(p.name for p in CASES.iterdir()))
def test_vessfire_deck_import_reproduces_the_deck(case_id):
    deck = read_deck(CASES / case_id)
    case, warnings = case_from_vessfire_deck(CASES / case_id)
    assert case.validate() == [], case.validate()
    assert any("material" in w for w in warnings)
    got = case.to_model_case()
    for k, v in deck["seg"].items():
        if k in ("material", "tag", "bdv_loc", "orientation"):
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                assert got["seg"][k][kk] == pytest.approx(vv, rel=1e-12), (k, kk)
        elif v is not None:
            assert got["seg"][k] == pytest.approx(v, rel=1e-12), k
    np.testing.assert_allclose(got["hl"]["series"], deck["hl"]["series"])
    for k in ("xi_start", "xi_end", "circ_deg", "attack_deg"):
        assert got["hl"][k] == deck["hl"][k]


def test_imported_case_runs_like_the_deck(steel):
    deck = read_deck(CASES / "M07-0004")
    case, _ = case_from_vessfire_deck(CASES / "M07-0004")
    opt = VesselFireOptions(t_end=120.0)
    a, _ = VesselFireModel(deck, opt, steel).run()
    b, _ = VesselFireModel(case.to_model_case(), opt, steel).run()
    pd.testing.assert_frame_equal(a, b, rtol=1e-9)


def test_run_case_progress_and_cancel():
    c = VesselCase()
    c.run.t_end_s, c.run.output_s = 60.0, 10.0
    seen = []
    res = run_case(c, progress=lambda t, T: seen.append((t, T)))
    assert seen[-1] == (60.0, 60.0) and len(res.series) == 7
    assert not res.failures.empty and res.runtime_s >= 0
    calls = []
    with pytest.raises(SimulationCancelled, match="t = 4 s"):
        run_case(c, cancel=lambda: calls.append(1) or len(calls) > 4)   # cancel at step 5
    with pytest.raises(ValueError, match="invalid case"):
        c.vessel.wall_m = -1
        run_case(c)
