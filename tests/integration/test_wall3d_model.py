"""The vessel model with the 3-D wall (wall_model="3d")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.coupling import VesselFireOptions, simulate
from fahts.rupture import evaluate
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"
COARSE = dict(wall_model="3d", wall3d_n_theta=36, wall3d_n_length=12, wall3d_n_radial=4)


def test_uniform_fire_agrees_with_1d_and_conserves_energy(steel):
    """Uniform fire: no lateral driving force, the 3-D wall must reproduce the 1-D one."""
    case = read_case(CASES / "M06-0003")                  # H2, 150 kW/m2, 60 mm
    a, _ = simulate(case, VesselFireOptions(t_end=300.0), steel)
    b, meta = simulate(case, VesselFireOptions(t_end=300.0, **COARSE), steel)
    assert abs(b.energy_err_pct.iloc[-1]) < 0.1                   # secant capacity: exact
    assert b.P_bara.iloc[-1] == pytest.approx(a.P_bara.iloc[-1], rel=5e-3)
    assert b.T_gas_C.iloc[-1] == pytest.approx(a.T_gas_C.iloc[-1], abs=2.0)
    assert b.background_T_mean_C.iloc[-1] == pytest.approx(a.background_T_mean_C.iloc[-1],
                                                           abs=5.0)
    w = meta["wall3d"]
    assert w["T"].shape == (len(b), w["mesh"].n_nodes) and len(meta["x_nodes"]) == 5


def test_jet_fire_hot_spot_drives_rupture(steel):
    case = read_case(CASES / "M06-0070")                  # LPG, jet on top
    ts, meta = simulate(case, VesselFireOptions(t_end=300.0, **COARSE), steel)
    assert abs(ts.energy_err_pct.iloc[-1]) < 0.1
    # the hot spot is the hottest region and in the jet zone
    assert (ts.hot_T_mean_C >= ts.peak_T_mean_C - 1e-9).all()
    assert ts.hot_T_mean_C.iloc[-1] > ts.background_T_mean_C.iloc[-1] + 20.0
    assert ts.T_mean_hot_C.iloc[-1] == pytest.approx(ts.hot_T_mean_C.iloc[-1])
    _ss, fail = evaluate(ts, case, steel, x_nodes=meta["x_nodes"])   # picks "hot"
    assert len(fail) and np.isfinite(ts.filter(like="hot_T").to_numpy()).all()
