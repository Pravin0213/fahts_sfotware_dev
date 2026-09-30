"""The vessel model's wall columns span the case's real wall thickness (known issue #1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.coupling import VesselFireModel, VesselFireOptions
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[2] / "regression" / "process" / "cases"


@pytest.mark.parametrize("case_id, t", [("M06-0003", 0.06), ("M04-0003", 0.1)])
def test_wall_columns_use_case_thickness(steel, case_id, t):
    case = read_case(CASES / case_id)
    assert case["seg"]["t"] == t
    model = VesselFireModel(case, VesselFireOptions(t_end=1.0), steel)
    for col in model.cols.values():
        assert col.R[-1] - col.R[0] == pytest.approx(t, rel=1e-12)
    np.testing.assert_allclose(model.x_nodes[-1], t)


def test_explicit_nodes_override(steel, caplog):
    case = read_case(CASES / "M06-0003")
    nodes = tuple(np.linspace(0.0, 0.105, 12))
    model = VesselFireModel(case, VesselFireOptions(t_end=1.0, wall_nodes=nodes), steel)
    assert model.x_nodes[-1] == pytest.approx(0.105)
    assert "wall nodes end at" in caplog.text
