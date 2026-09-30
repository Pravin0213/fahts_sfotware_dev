"""fahts.wall.column_1d + fahts.fire must reproduce legacy heat_transfer bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from fahts import fire
from fahts.wall.column_1d import WallColumn
from tests.legacy_ref import legacy

NODES_105MM = legacy("heat_transfer").VESSFIRE_LOG_NODES_105MM

pytest.importorskip("CoolProp")


def _run(ht_mod, vessel_mod, mat, mode, n=300):
    """Heat a wall column: fire for 200 s then ambient (wind) cooling."""
    amb = vessel_mod.AmbientAuto(293.15, 0.85, 2.12, -0.5)
    f = ht_mod.GuidelineFire(eps_flame=1.0, eps_surf=0.7, h_flame=25.0, T_ref=293.15,
                             t_air_mode=mode)
    bc = ht_mod.FireBC(f, [0.0, 200.0, 201.0, 1e6], [150e3, 150e3, 0.0, 0.0], after=amb)
    col = ht_mod.WallColumn(mat, 1.0, NODES_105MM, bc, 293.15)
    out = []
    for k in range(n):
        col.step(1.0, k + 0.5, 300.0 + 0.5 * k, 50.0 + k)
        out.append((col.T.copy(), col.q_net_out, col.q_rad_out, col.q_in, col.T_flame))
    return out, col.energy(), col.T_mean()


@pytest.mark.parametrize("mode", ["blackbody", "balance"])
def test_wall_column_under_fire_and_ambient(steel, legacy_steel, mode):
    class New:                        # adapter: fahts modules under the legacy names
        GuidelineFire, FireBC, WallColumn = fire.GuidelineFire, fire.FireBC, WallColumn
        AmbientAuto = fire.AmbientAuto
    new = _run(New, New, steel, mode)
    old = _run(legacy("heat_transfer"), legacy("vessel"), legacy_steel, mode)
    for a, b in zip(new[0], old[0]):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert new[1:] == old[1:]


def test_ambient_h_forced_and_natural():
    old = legacy("vessel")
    for spec in (-0.5, -5.0, -100.0, 12.0):
        for Ts in (250.0, 293.15, 400.0, 900.0):
            assert fire.h_ambient(Ts, 293.15, 2.1, spec) == old.h_ambient(Ts, 293.15, 2.1, spec)
