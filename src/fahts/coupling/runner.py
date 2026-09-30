"""Run a ``VesselCase``: model + rupture evaluation, with progress and cancellation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from fahts.coupling.vessel_case import VesselCase
from fahts.coupling.vessel_fire_1d import VesselFireModel
from fahts.rupture import evaluate


@dataclass
class CaseResult:
    case: VesselCase
    series: pd.DataFrame  # model output rows (see VesselFireModel._record)
    stress: pd.DataFrame  # through-wall stress / allowable series (rupture.stress_series)
    failures: pd.DataFrame  # failure times per variant / basis / criterion
    meta: dict = field(default_factory=dict)
    runtime_s: float = 0.0

    @property
    def rupture_tresca_s(self) -> float | None:
        return self.meta.get("rupture", {}).get("Tresca")

    @property
    def rupture_von_mises_s(self) -> float | None:
        return self.meta.get("rupture", {}).get("vonMises")


def run_case(case: VesselCase, progress=None, cancel=None) -> CaseResult:
    """Validate and run one case. Raises ValueError listing all input errors."""
    errors = case.validate()
    if errors:
        raise ValueError("invalid case:\n  " + "\n  ".join(errors))
    t0 = time.perf_counter()
    model_case = case.to_model_case()
    mat = case.vessel.material.steel_table()
    ts, meta = VesselFireModel(model_case, case.model_options(), mat).run(progress, cancel)
    stress, failures = evaluate(ts, model_case, mat, x_nodes=meta["x_nodes"])
    return CaseResult(case, ts, stress, failures, meta, round(time.perf_counter() - t0, 1))
