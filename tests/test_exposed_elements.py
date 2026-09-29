"""GUI pre-run exposure count must include shells (plates) — regression 2026-09-27:
a RadiationBall between two plates showed 'No exposed elements' because only beams
were screened."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fahts.core.heat.solver.analysis_runner import exposed_analysis_element_ids, run_analysis
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.results.analysis_config import AnalysisConfig

ROOT = Path(__file__).resolve().parents[1]


def _ball(active: bool = True) -> RadiationBall:
    return RadiationBall(name="B", center=np.array([1.0, 1.0, 0.5]), radius=0.4,
                         flux=100e3, active=active)


def test_plates_counted_for_radiation_ball():
    model = read_usfos_fem(ROOT / "examples" / "models" / "parallel_plates.fem")
    assert not model.elements and len(model.shell_elements) == 32
    assert exposed_analysis_element_ids(model, [_ball()]) == set(model.shell_elements)
    assert exposed_analysis_element_ids(model, [_ball(active=False)]) == set()


def test_plates_run_with_radiation_ball():
    model = read_usfos_fem(ROOT / "examples" / "models" / "parallel_plates.fem")
    res = run_analysis(model, [_ball()], AnalysisConfig(t_end=300.0, dt=30.0, output_dt=300.0))
    assert set(res.element_ids) == set(model.shell_elements)
    T = res.T_centroid[-1]
    assert np.all(np.isfinite(T)) and T.max() > 25.0 and T.min() >= 20.0 - 1e-9
    # symmetric set-up: both plates heat equally (ball midway between them)
    z = {e: np.mean([model.nodes[n].z for n in model.shell_elements[e].nodes])
         for e in res.element_ids}
    lo = np.mean([T[i] for i, e in enumerate(res.element_ids) if z[e] < 0.5])
    hi = np.mean([T[i] for i, e in enumerate(res.element_ids) if z[e] > 0.5])
    # identical up to the Monte Carlo noise of the exchange view factors (0.002 K here)
    assert abs(lo - hi) < 1e-4 * max(lo, hi)
