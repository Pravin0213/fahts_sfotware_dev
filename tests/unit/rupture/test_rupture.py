"""fahts.rupture: FE vs closed-form check, and bit-identical equivalence with legacy stress.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fahts.rupture import evaluate, membrane_stresses
from fahts.rupture.gps_fe import GeneralizedPlaneStrainFE
from fahts.rupture.stress_solutions import lame, thermal_tg
from tests.legacy_ref import legacy

NODES_105MM = legacy("heat_transfer").VESSFIRE_LOG_NODES_105MM


def test_fe_matches_closed_form_for_constant_E():
    """Pressure (Lame) + thermal (T&G) with constant E: FE reproduces the closed forms
    (legacy stress._selftest)."""
    a, b, p, s_ext = 1.29, 1.395, 20.0, 30.0
    Fz = p * math.pi * a**2 + s_ext * math.pi * (b**2 - a**2)
    rT = np.linspace(a, b, 12)
    e = 1.3e-5 * 300 * np.log(rT / a) / np.log(b / a)
    f = GeneralizedPlaneStrainFE(a, b, 200).solve(lambda r: 2e5, lambda r: np.interp(r, rT, e),
                                                  p, Fz)
    rl = np.array([a, 0.5 * (a + b), b])
    lr, lt, lz = lame(p, a, b, rl, s_ext)
    tr_, tt_, tz_ = thermal_tg(rT, e, rl, 2e5)
    closed = np.array([lr + tr_, lt + tt_, lz + tz_])
    assert np.abs(np.array(f(rl)) - closed).max() < 0.5          # MPa, on ~300 MPa stresses


def _series(n=40):
    """Synthetic model output: pressure rise and a heating wall with a through-wall gradient."""
    t = np.arange(n) * 30.0
    df = pd.DataFrame({"Time": t, "P_bara": 100.0 + 0.8 * np.arange(n)})
    for col, rate in (("background", 25.0), ("wet", 3.0)):
        T_in = 20.0 + rate * np.arange(n)
        for i in range(12):
            df[f"{col}_T{i+1}_C"] = T_in + 0.06 * rate * i * np.arange(n) / 12
        df[f"{col}_T_mean_C"] = df[[f"{col}_T{i+1}_C" for i in range(12)]].mean(axis=1)
    return df


def test_equivalence_with_legacy(steel, legacy_steel):
    case = {"seg": dict(D=2.0, t=0.06, strength_mpa=490.0, stress_factor=1.0,
                        stress_type="U", ext_long_mpa=30.0, material="synthetic")}
    py = _series()
    old = legacy("stress")
    ss_old = old.stress_series(py, case["seg"], legacy_steel, "background")
    ft_old = pd.DataFrame(old.failure_times(ss_old))
    ss_new, ft_new = evaluate(py, case, steel, x_nodes=NODES_105MM)
    pd.testing.assert_frame_equal(ss_new, ss_old, check_exact=True)
    pd.testing.assert_frame_equal(ft_new, ft_old, check_exact=True)
    assert ft_new.t_fail_s.notna().any()                          # the test does reach failure


def test_membrane_stresses_equivalence():
    old = legacy("vessel")
    for P in (1e5, 50e5, 150e5):
        assert membrane_stresses(P, 2.0, 0.06, 30.0) == old.stresses(P, 2.0, 0.06, 30.0)
