"""The vessel model's 3-D wall spans the case's real wall thickness (known issue #1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.coupling import VesselFireModel, VesselFireOptions
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[2] / "regression" / "process" / "cases"
COARSE = dict(wall3d_n_theta=12, wall3d_n_length=4, wall3d_n_radial=3)


@pytest.mark.parametrize("case_id, t", [("M06-0003", 0.06), ("M04-0003", 0.1)])
def test_wall_uses_case_thickness(steel, case_id, t):
    case = read_case(CASES / case_id)
    assert case["seg"]["t"] == t
    model = VesselFireModel(case, VesselFireOptions(t_end=1.0, **COARSE), steel)
    r = model.wall3d.mesh.r
    assert r[-1] - r[0] == pytest.approx(t, rel=1e-12)
    assert r[0] == pytest.approx(case["seg"]["D"] / 2)
    np.testing.assert_allclose(model.x_nodes[-1], t)
