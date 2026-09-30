"""The coupled vessel model conserves energy: wall + contents + outflow = fire heat in.

Known issue #1 (wall grid from a 105 mm shell for every vessel) broke this for other wall
thicknesses (56 % error on the 60 mm validation vessel), so it guards the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fahts.coupling import VesselFireOptions, simulate
from validation.vessfire.input_deck import read_case

pytest.importorskip("CoolProp")
CASES = Path(__file__).resolve().parents[1] / "regression" / "process" / "cases"


@pytest.mark.parametrize("case_id", ["M06-0003", "M07-0004", "M04-0003"])
def test_energy_balance_closes_and_converges(steel, case_id):
    """H2 closed fire (60 mm wall), CH4 fire + BDV, H2 cold blowdown (100 mm wall).

    The residual is the first-order time error of the wall's lagged cp (issue #6): small,
    and it shrinks in proportion to dt. The pre-fix grid gave 56 % on M06-0003.
    """
    case = read_case(CASES / case_id)
    err = {}
    for dt in (1.0, 0.25):
        ts, _ = simulate(case, VesselFireOptions(t_end=300.0, dt=dt), steel)
        err[dt] = abs(ts.energy_err_pct.iloc[-1])
    assert err[1.0] < 1.5
    assert err[0.25] < max(err[1.0] / 3.0, 1e-3)
