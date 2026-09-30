"""The vessel model with its 3-D (Hex8 solid) wall."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.coupling import VesselFireOptions, simulate
from fahts.rupture import evaluate
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"
COARSE = dict(wall3d_n_theta=36, wall3d_n_length=12, wall3d_n_radial=4)


def test_uniform_fire_mesh_independent_and_conserves_energy(steel):
    """Uniform fire: a coarse and a 2x finer wall mesh give the same vessel response
    (full mesh study: validation/vessfire/reports/wall3d_mesh.md)."""
    case = read_case(CASES / "M06-0003")                  # H2, 150 kW/m2, 60 mm
    fine = dict(wall3d_n_theta=72, wall3d_n_length=24, wall3d_n_radial=8)
    a, _ = simulate(case, VesselFireOptions(t_end=300.0, **fine), steel)
    b, meta = simulate(case, VesselFireOptions(t_end=300.0, **COARSE), steel)
    assert abs(b.energy_err_pct.iloc[-1]) < 0.1                   # secant capacity: exact
    assert b.P_bara.iloc[-1] == pytest.approx(a.P_bara.iloc[-1], rel=5e-3)
    assert b.T_gas_C.iloc[-1] == pytest.approx(a.T_gas_C.iloc[-1], abs=2.0)
    assert b.background_T_mean_C.iloc[-1] == pytest.approx(a.background_T_mean_C.iloc[-1],
                                                           abs=5.0)
    assert b.background_T_out_C.iloc[-1] > b.background_T_in_C.iloc[-1] + 20.0  # gradient
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
