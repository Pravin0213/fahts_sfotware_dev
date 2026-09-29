"""
Phase 3E.8 — USFOS benchmark comparison.

Benchmark case
--------------
Model  : model_t1.fem   (IHPROFIL + PIPE + BOX + ECCENT + shells)
Source : RadiationBall  center=(343,484,64)m, radius=5m, flux=350kW/m²
USFOS  : validation/usfos/reference/fahts_beltemp.fem  (15 min, 1-min output steps)

NOTE: The reference USFOS run used the old two-zone USERFLUX calibration
(r1=5m/flux1=350kW, r2=100m/flux2=1.5kW). FAHTS RadiationBall has since moved
to a single-zone exact point-to-sphere model (radius, flux) — see rad_ball.py.
This test uses radius=r1, flux=flux1; outer-zone (far-field) quantitative
agreement with the old r2/flux2 calibration point is expected to differ and
has not been re-tuned — flagged as a follow-up, not a regression.

Known physics differences (FAHTS vs USFOS)
-------------------------------------------
1. Re-radiation:
   USFOS USERFLUX applies net heat flux including Stefan-Boltzmann re-radiation from the
   steel surface, so net q decreases as steel heats up and plateaus at radiation equilibrium.
   FAHTS RadiationBall currently applies the prescribed flux as-is (no re-radiation term),
   causing inner-zone elements to overheat without limit.

2. 2-D vs 3-D:
   FAHTS uses a 2-D cross-section FEM (no axial conduction).
   USFOS uses a full 3-D beam FEM; axial conduction along the beam length is included.

3. Material properties:
   FAHTS  — EN 1993-1-2 Annex C piecewise-linear tables (cp(20°C)=440, k(20°C)=54).
   USFOS  — thermpar × tempdepy multiplier tables (cp_ref=510×0.792=404, k_ref=50×1.084=54.2 at 20°C).
   Both use similar k(T) but FAHTS cp is ~9% higher at low T, so FAHTS heats up slightly slower.

4. Element-level flux distribution:
   FAHTS applies the zone flux to the element midpoint distance.
   USFOS may compute flux at individual nodes/integration points (3-D solid mesh).

Validation strategy
-------------------
- Quick tests (always run): parsing sanity, physics ordering checks.
- Full comparison (run_full_comparison function): 15-min run on representative elements,
  tabular FAHTS vs USFOS report.

Run the full comparison manually:
    python -m pytest tests/test_usfos_benchmark.py::run_full_comparison -s
or:
    python tests/test_usfos_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_MODEL   = _ROOT / "examples" / "models" / "model_t1.fem"
_BELTEMP = _ROOT / "validation" / "usfos" / "reference" / "fahts_beltemp.fem"
_BALL_CENTER = np.array([343.0, 484.0, 64.0])
_RADIUS, _FLUX = 5.0, 350_000.0


def _make_ball():
    from fahts.core.heat.sources.rad_ball import RadiationBall
    return RadiationBall(
        name="usfos_benchmark",
        center=_BALL_CENTER,
        radius=_RADIUS, flux=_FLUX,
    )


def _load_model():
    from fahts.core.io.usfos_reader import read_usfos_fem
    return read_usfos_fem(_MODEL)


def _load_reference():
    from fahts.core.io.beltemp_parser import parse_beltemp, cumulative_temperatures
    recs = parse_beltemp(_BELTEMP)
    return cumulative_temperatures(recs)


# ── Reference data sanity ─────────────────────────────────────────────────────

class TestBeltempParsing:
    """Verify the USFOS reference file loads correctly."""

    def test_element_count(self):
        ct = _load_reference()
        # USFOS outputs results only for exposed elements.
        # The reference fahts_beltemp.fem contains 1802 entries (out of
        # 1644 beams + 412 shells = 2056 total in model_t1.fem).
        assert len(ct) == 1802

    def test_time_steps(self):
        ct = _load_reference()
        # Pick any element — should have 16 entries (t=0 + 15 min)
        eid = next(iter(ct))
        times, _, _, _ = ct[eid]
        assert len(times) == 16
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(15.0)

    def test_inner_zone_reference_temperatures_reasonable(self):
        """Inner-zone elements should reach high temperatures (>350°C) at 15 min.
        eid=65 reaches ~399°C and eid=300 ~379°C in the USFOS reference;
        threshold set to 350°C to cover all four inner-zone elements.
        """
        ct = _load_reference()
        inner_eids = [65, 66, 296, 300]  # dist ≤ 5m, verified from model
        for eid in inner_eids:
            times, T_mean, _, _ = ct[eid]
            T15 = float(np.interp(15.0, times, T_mean))
            assert T15 > 350.0, f"eid={eid}: expected T@15min > 350°C, got {T15:.1f}"

    def test_outer_zone_reference_temperatures_lower(self):
        """Outer-zone elements should be cooler than inner-zone at 15 min."""
        ct = _load_reference()
        T_inner = float(np.interp(15.0, *ct[65][:2]))
        T_outer_near = float(np.interp(15.0, *ct[68][:2]))  # dist~5.2m
        T_outer_far  = float(np.interp(15.0, *ct[400][:2]))  # dist~12.8m
        assert T_inner > T_outer_near > T_outer_far

    def test_monotonic_reference_temperatures(self):
        """
        Reference temperatures must be broadly non-decreasing.
        Inner-zone elements may dip slightly (<0.1°C) as they approach
        Stefan-Boltzmann equilibrium — allow a small tolerance.
        """
        ct = _load_reference()
        for eid in [65, 296, 68, 100, 400]:
            _, T_mean, _, _ = ct[eid]
            diffs = np.diff(T_mean)
            assert np.all(diffs >= -0.15), (
                f"Non-monotonic USFOS T for eid={eid}: min_diff={diffs.min():.3f}"
            )


# ── FAHTS quick physics checks ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def quick_result():
    """
    Run FAHTS on 4 representative elements for 5 minutes.
    dt=10s, output every 60s.  Completes in ~2 seconds.
    """
    from fahts.core.heat.solver.analysis_runner import run_analysis
    from fahts.core.results.analysis_config import AnalysisConfig

    model = _load_model()
    ball  = _make_ball()
    # inner zone: 65 (PIPE dist=3.66m), 296 (IHPROFIL dist=2.69m)
    # outer zone: 400 (IHPROFIL dist=12.78m), 500 (PIPE dist=10.09m)
    cfg = AnalysisConfig(
        t_end=300.0, dt=10.0, output_dt=60.0,
        n_layers=2, element_ids=[65, 296, 400, 500],
    )
    return run_analysis(model, [ball], cfg)


class TestFAHTSPhysics:
    """Physics sanity checks on the FAHTS 5-minute run."""

    def test_all_elements_solved(self, quick_result):
        assert set(quick_result.element_ids) == {65, 296, 400, 500}

    def test_temperature_increases_monotonically(self, quick_result):
        for eid in quick_result.element_ids:
            idx = quick_result.element_ids.index(eid)
            T = quick_result.T_centroid[:, idx]
            diffs = np.diff(T)
            assert np.all(diffs >= 0), f"eid={eid}: non-monotonic T"

    def test_initial_temperature_is_ambient(self, quick_result):
        for eid in quick_result.element_ids:
            idx = quick_result.element_ids.index(eid)
            T0 = quick_result.T_centroid[0, idx]
            assert T0 == pytest.approx(20.0, abs=0.1), f"eid={eid}: T0={T0}"

    def test_inner_zone_hotter_than_outer_zone(self, quick_result):
        """At every time step, inner-zone elements must be hotter than outer-zone."""
        def T_at(eid, step):
            idx = quick_result.element_ids.index(eid)
            return quick_result.T_centroid[step, idx]

        for step in range(1, quick_result.n_steps):
            T_inner_max = max(T_at(65, step), T_at(296, step))
            T_outer_max = max(T_at(400, step), T_at(500, step))
            assert T_inner_max > T_outer_max, (
                f"step={step}: inner max={T_inner_max:.1f} not > outer max={T_outer_max:.1f}"
            )

    def test_inner_zone_exceeds_100c_in_1min(self, quick_result):
        """Inner-zone element at dist=2.7m, flux=350kW/m² must exceed 100°C in 60 seconds."""
        idx = quick_result.element_ids.index(296)
        T60 = float(np.interp(60.0, quick_result.times, quick_result.T_centroid[:, idx]))
        assert T60 > 100.0, f"T@60s={T60:.1f}°C"

    def test_outer_zone_moderate_heating(self, quick_result):
        """Outer-zone elements (flux=1500 W/m²) must heat above ambient in 5 minutes.

        Threshold is 22°C (2°C above ambient) to account for re-radiation (ε·σ·T⁴)
        which reduces net flux from 1500 → ~1208 W/m² at 20°C.  High-mass PIPE
        sections heat slowly but must still show measurable temperature rise.
        """
        for eid in [400, 500]:
            idx = quick_result.element_ids.index(eid)
            T_end = quick_result.T_centroid[-1, idx]
            assert T_end > 22.0, f"eid={eid}: T@5min={T_end:.1f}°C — outer zone barely heated"


# ── Quantitative comparison report ───────────────────────────────────────────

def _usfos_at(ct, eid, t_min):
    if eid not in ct:
        return float("nan")
    times, T_mean, _, _ = ct[eid]
    return float(np.interp(t_min, times, T_mean))


def _fahts_at(result, eid, t_s):
    if eid not in result.element_ids:
        return float("nan")
    idx = result.element_ids.index(eid)
    return float(np.interp(t_s, result.times, result.T_centroid[:, idx]))


def run_full_comparison(print_report: bool = True) -> dict:
    """
    Run FAHTS for 15 minutes on representative elements and compare to USFOS.

    Parameters
    ----------
    print_report : If True, print the comparison table to stdout.

    Returns
    -------
    dict with keys:
        'result'       : TemperatureField from FAHTS
        'ct'           : cumulative USFOS temperatures
        'sample_eids'  : element IDs analysed
    """
    from fahts.core.heat.solver.analysis_runner import run_analysis
    from fahts.core.results.analysis_config import AnalysisConfig

    model = _load_model()
    ball  = _make_ball()
    ct    = _load_reference()

    # Representative sample: inner zone (2 PIPE, 1 I), outer zone (2 I, 1 PIPE)
    sample_eids = [65, 66, 296, 68, 100, 400, 500]

    cfg = AnalysisConfig(
        t_end=900.0,       # 15 min in seconds
        dt=3.0,            # matches USFOS nStep=300/15min
        output_dt=60.0,    # 1-min output steps
        n_layers=4,        # matches USFOS MESHBOX/MESHIPRO 4-layer mesh
        element_ids=sample_eids,
    )
    result = run_analysis(model, [ball], cfg)

    if print_report:
        _print_comparison(result, ct, sample_eids, model)

    return {"result": result, "ct": ct, "sample_eids": sample_eids}


def _print_comparison(result, ct, sample_eids, model):
    from fahts.core.io.usfos_reader import read_usfos_fem

    lines = [
        "",
        "=" * 76,
        "FAHTS vs USFOS Benchmark Comparison",
        "Model: model_t1.fem   Source: RadiationBall center=(343,484,64)m",
        f"       radius={_RADIUS}m  flux={_FLUX/1e3:.0f} kW/m²",
        "=" * 76,
        "",
        "Known physics differences:",
        "  [A] USFOS applies Stefan-Boltzmann re-radiation; FAHTS does not.",
        "      → FAHTS inner-zone T rises without bound; USFOS plateaus.",
        "  [B] USFOS 3-D beam FEM includes axial conduction; FAHTS 2-D does not.",
        "  [C] Material: FAHTS EN1993-1-2 cp(20°C)=440 vs USFOS cp=404 J/(kg·K).",
        "",
    ]

    # Per-element comparison
    t_mins = [1, 3, 5, 10, 15]
    for eid in sample_eids:
        if eid not in result.element_ids:
            continue
        e = model.elements.get(eid)
        if e is None:
            continue
        mid = e.midpoint(model.nodes)
        d = float(np.linalg.norm(mid - _BALL_CENTER))
        sec = model.sections.get(e.geom_id)
        zone = "INNER" if d <= _RADIUS else "OUTER"

        lines.append(
            f"Element {eid:4d} | {type(sec).__name__:12s} | dist={d:6.2f}m | {zone} zone"
        )
        lines.append(f"  {'t(min)':>7} | {'FAHTS(°C)':>10} | {'USFOS(°C)':>10} | {'diff':>8} | {'rel%':>8}")
        lines.append(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        for t_min in t_mins:
            T_f = _fahts_at(result, eid, t_min * 60)
            T_u = _usfos_at(ct, eid, float(t_min))
            diff = T_f - T_u
            rel  = 100 * diff / T_u if abs(T_u) > 1 else float("nan")
            lines.append(
                f"  {t_min:7d} | {T_f:10.1f} | {T_u:10.1f} | {diff:+8.1f} | {rel:+8.1f}%"
            )
        lines.append("")

    lines += [
        "Summary of main discrepancies:",
        "  • Inner zone:  FAHTS T rises well above USFOS (~500-3000°C diff at 15 min).",
        "    Root cause: FAHTS RadiationBall does not apply Stefan-Boltzmann re-radiation.",
        "    Fix needed: Add ε × σ × T_steel^4 re-radiation term to RadiationBall BC.",
        "  • Outer zone:  FAHTS T lower than USFOS at all time steps.",
        "    Likely cause: Flux magnitude or boundary condition difference in USFOS.",
        "    Needs further investigation of USFOS USERFLUX outer-zone BC.",
        "=" * 76,
    ]

    print("\n".join(lines))


@pytest.mark.slow
def test_full_comparison_runs():
    """Smoke test: full 15-min comparison must complete without error (marked slow)."""
    data = run_full_comparison(print_report=False)
    result = data["result"]
    assert result.n_elements == len(data["sample_eids"])
    assert result.n_steps == 16  # t=0 + 15 output steps


# ── Entry point for standalone run ───────────────────────────────────────────

if __name__ == "__main__":
    run_full_comparison(print_report=True)
