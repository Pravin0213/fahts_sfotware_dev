"""Process-model regression: the product code must reproduce the golden outputs.

A failure means results changed. If the change is unintended, find out why. If it is a
deliberate physics change, regenerate the goldens in the same commit and say why
(``python -m tests.regression.process.generate``). Never loosen RTOL to hide a change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.regression.process import harness as h

pytest.importorskip("CoolProp")
pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not h.have_material_db(),
                       reason=f"material DB not present: {h.MATERIAL_DB}"),
]

# Relative tolerance on every numeric column, scaled by the column's magnitude.
# A pure refactor can reorder floating-point operations; the flash iterations can
# amplify that a little, but anything above this is a behaviour change.
RTOL = 1e-6


def _compare_frames(new: pd.DataFrame, ref: pd.DataFrame, what: str) -> None:
    assert list(new.columns) == list(ref.columns), f"{what}: columns differ"
    assert len(new) == len(ref), f"{what}: {len(new)} rows vs golden {len(ref)}"
    for col in ref.columns:
        r, n = ref[col], new[col]
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(n):
            rv, nv = r.to_numpy(float), n.to_numpy(float)
            scale = np.nanmax(np.abs(rv)) if np.isfinite(rv).any() else 0.0
            np.testing.assert_allclose(nv, rv, rtol=RTOL, atol=RTOL * scale + 1e-12,
                                       equal_nan=True, err_msg=f"{what}: column {col!r}")
        else:
            assert n.astype(str).tolist() == r.astype(str).tolist(), \
                f"{what}: column {col!r} differs"


@pytest.mark.parametrize("impl", h.GOLDEN_IMPLEMENTATIONS)
@pytest.mark.parametrize("gr", h.RUNS, ids=lambda r: r.name)
def test_matches_golden(gr: h.GoldenRun, impl: str) -> None:
    ts_path, rup_path = h.golden_paths(gr)
    if not ts_path.exists():
        pytest.fail(f"golden missing: {ts_path.name} (python -m tests.regression.process.generate)")
    ts, failures = h.run(gr, impl)
    # round-trip through CSV so dtypes match the stored goldens exactly
    ts = pd.read_csv(pd.io.common.StringIO(ts.to_csv(index=False, float_format="%.17g")))
    failures = pd.read_csv(pd.io.common.StringIO(
        failures.to_csv(index=False, float_format="%.17g")))
    _compare_frames(ts, pd.read_csv(ts_path), f"{gr.name} time series")
    _compare_frames(failures, pd.read_csv(rup_path), f"{gr.name} rupture")
