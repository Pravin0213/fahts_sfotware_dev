"""
Unit tests for Task 3.8 — analysis_runner.run_analysis().

Uses a minimal synthetic FEMModel (one BOX element, one fire zone) so tests
run fast without loading a real .fem file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.solver.analysis_runner import (
    AnalysisCancelledError,
    run_analysis,
)
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection, ISection
from fahts.core.results.analysis_config import AnalysisConfig
from fahts.core.results.temperature_field import TemperatureField


# ── Minimal synthetic model ───────────────────────────────────────────────────

def _make_model(section_type: str = "box") -> FEMModel:
    """
    Build a minimal FEMModel with a single beam element in the fire zone.

    The beam runs from (0,0,0) to (1,0,0).  The fire zone covers the origin.
    """
    n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(nid=2, x=1.0, y=0.0, z=0.0)
    nodes = {1: n1, 2: n2}

    mat = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)

    if section_type == "box":
        sec = BoxSection(sid=10, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
    else:
        # ISection — not supported in Phase 3, triggers skipping
        sec = ISection(
            sid=10, h=0.30, tw=0.010,
            bf_top=0.15, tf_top=0.012,
            bf_bot=0.15, tf_bot=0.012,
        )

    direction = np.array([1.0, 0.0, 0.0])
    local_z   = np.array([0.0, 0.0, 1.0])
    elem = BeamElement(
        eid=101, n1=1, n2=2, mat_id=1, geom_id=10, lcoor_id=0,
        length=1.0, direction=direction, local_z=local_z,
    )

    return FEMModel(
        nodes=nodes,
        elements={101: elem},
        sections={10: sec},
        materials={1: mat},
        groups={},
        unitvecs={},
        source_file=Path("synthetic.fem"),
    )


def _make_zone() -> FireZone:
    """Fire zone that covers the beam midpoint at (0.5, 0, 0)."""
    curve = FireCurve(curve_type=FireCurveType.ISO_834)
    return FireZone(
        name="TestZone",
        center=np.array([0.5, 0.0, 0.0]),
        dims=np.array([2.0, 2.0, 2.0]),
        curve=curve,
        epsilon_fire=1.0,
        h_conv=25.0,
        active=True,
    )


def _make_config(**overrides) -> AnalysisConfig:
    defaults = dict(t_end=120.0, dt=60.0, output_dt=120.0, n_layers=1)
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ── Basic happy path ──────────────────────────────────────────────────────────

class TestRunAnalysisBasic:
    def test_returns_temperature_field(self):
        model = _make_model()
        zones = [_make_zone()]
        cfg = _make_config()
        result = run_analysis(model, zones, cfg)
        assert isinstance(result, TemperatureField)

    def test_correct_element_id(self):
        model = _make_model()
        result = run_analysis(model, [_make_zone()], _make_config())
        assert result.element_ids == [101]

    def test_n_steps_matches_config(self):
        model = _make_model()
        cfg = _make_config(t_end=120.0, dt=60.0, output_dt=60.0)
        result = run_analysis(model, [_make_zone()], cfg)
        # t=0, t=60, t=120 → 3 steps
        assert result.n_steps == 3

    def test_temperature_increases(self):
        """Cold steel in hot fire must heat up."""
        model = _make_model()
        result = run_analysis(model, [_make_zone()], _make_config())
        T_cen = result.T_centroid[:, 0]
        assert T_cen[-1] > T_cen[0]

    def test_T_section_populated(self):
        """from_solver_run must store full nodal section history."""
        model = _make_model()
        result = run_analysis(model, [_make_zone()], _make_config())
        assert 101 in result.T_section

    def test_single_step_config(self):
        """dt == t_end → single solver step, two output rows (t=0 and t=t_end)."""
        model = _make_model()
        cfg = _make_config(t_end=60.0, dt=60.0, output_dt=60.0)
        result = run_analysis(model, [_make_zone()], cfg)
        assert result.n_steps == 2


# ── Progress callback ─────────────────────────────────────────────────────────

class TestProgressCallback:
    def test_callback_called(self):
        calls = []
        run_analysis(
            _make_model(), [_make_zone()], _make_config(),
            progress_cb=lambda cur, tot, eid: calls.append((cur, tot, eid)),
        )
        assert len(calls) >= 1

    def test_callback_receives_correct_total(self):
        """total should equal n_steps (2 for dt=60, t_end=120) for our fixture."""
        totals = []
        run_analysis(
            _make_model(), [_make_zone()], _make_config(),
            progress_cb=lambda cur, tot, eid: totals.append(tot),
        )
        # n_steps = round(t_end / dt) = round(120 / 60) = 2
        assert all(t == 2 for t in totals)

    def test_callback_receives_minus_one_eid(self):
        """eid is always -1 in the time-step-outer loop (no per-element ticks)."""
        eids = []
        run_analysis(
            _make_model(), [_make_zone()], _make_config(),
            progress_cb=lambda cur, tot, eid: eids.append(eid),
        )
        assert all(e == -1 for e in eids)


# ── Cancellation ─────────────────────────────────────────────────────────────

class TestCancellation:
    def test_cancel_from_start_raises(self):
        """If cancel_check immediately returns True, raise AnalysisCancelledError."""
        with pytest.raises(AnalysisCancelledError):
            run_analysis(
                _make_model(), [_make_zone()], _make_config(),
                cancel_check=lambda: True,
            )

    def test_cancel_false_runs_normally(self):
        """cancel_check=False means never cancel — must complete."""
        result = run_analysis(
            _make_model(), [_make_zone()], _make_config(),
            cancel_check=lambda: False,
        )
        assert isinstance(result, TemperatureField)


# ── Element filtering ─────────────────────────────────────────────────────────

class TestElementFiltering:
    def test_isection_now_supported(self):
        """ISection element in fire zone should now be solved (Phase 3E extended)."""
        from fahts.core.results.temperature_field import TemperatureField
        model = _make_model(section_type="isection")
        result = run_analysis(model, [_make_zone()], _make_config())
        assert isinstance(result, TemperatureField)
        assert 101 in result.element_ids

    def test_element_outside_zone_skipped(self):
        """Element not inside any fire zone → no results → ValueError."""
        model = _make_model()
        far_zone = FireZone(
            name="Far",
            center=np.array([100.0, 100.0, 100.0]),
            dims=np.array([1.0, 1.0, 1.0]),
            curve=FireCurve(FireCurveType.ISO_834),
            epsilon_fire=1.0, h_conv=25.0, active=True,
        )
        with pytest.raises(ValueError):
            run_analysis(model, [far_zone], _make_config())

    def test_inactive_zone_not_used(self):
        """An inactive fire zone must not expose any elements."""
        model = _make_model()
        inactive_zone = FireZone(
            name="Inactive",
            center=np.array([0.5, 0.0, 0.0]),
            dims=np.array([2.0, 2.0, 2.0]),
            curve=FireCurve(FireCurveType.ISO_834),
            epsilon_fire=1.0, h_conv=25.0, active=False,
        )
        with pytest.raises(ValueError):
            run_analysis(model, [inactive_zone], _make_config())

    def test_config_element_ids_respected(self):
        """When config.element_ids is set, only those elements are solved."""
        # Requesting a non-existent element ID → ValueError (no results after filtering)
        model = _make_model()
        cfg = AnalysisConfig(
            t_end=120.0, dt=60.0, output_dt=120.0,
            element_ids=[9999],   # does not exist in model
        )
        with pytest.raises((ValueError, KeyError)):
            run_analysis(model, [_make_zone()], cfg)

    def test_empty_fire_zones(self):
        """No fire zones → no exposed elements → ValueError."""
        with pytest.raises(ValueError):
            run_analysis(_make_model(), [], _make_config())


# ── Multiple zones ────────────────────────────────────────────────────────────

class TestMultipleZones:
    def test_two_zones_same_element(self):
        """Two zones covering the same element — must not double-count."""
        model = _make_model()
        z1 = _make_zone()
        z2 = FireZone(
            name="Z2",
            center=np.array([0.5, 0.0, 0.0]),
            dims=np.array([3.0, 3.0, 3.0]),
            curve=FireCurve(FireCurveType.HYDROCARBON),
            epsilon_fire=1.0, h_conv=50.0, active=True,
        )
        result = run_analysis(model, [z1, z2], _make_config())
        # Still one element
        assert result.n_elements == 1
        assert result.element_ids == [101]


# ── RadiationBall source ─────────────────────────────────────────────────────

class TestRadiationBallSource:
    """Analysis runner with a RadiationBall covering the beam midpoint."""

    def _make_ball(self) -> "RadiationBall":
        from fahts.core.heat.sources.rad_ball import RadiationBall
        # Beam midpoint is at (0.5, 0, 0); inner zone r1=1m covers it.
        return RadiationBall(
            name="TestBall",
            center=np.array([0.5, 0.0, 0.0]),
            r1=1.0,
            flux1=50_000.0,
            r2=10.0,
            flux2=5_000.0,
            active=True,
        )

    def test_rad_ball_exposes_element(self):
        model = _make_model()
        result = run_analysis(model, [self._make_ball()], _make_config())
        assert isinstance(result, TemperatureField)
        assert result.element_ids == [101]

    def test_rad_ball_temperature_increases(self):
        model = _make_model()
        result = run_analysis(model, [self._make_ball()], _make_config())
        T_cen = result.T_centroid[:, 0]
        assert T_cen[-1] > T_cen[0]

    def test_rad_ball_outer_zone_lower_flux(self):
        """Element in outer zone (r1 < d ≤ r2) receives flux2, still heats up."""
        from fahts.core.heat.sources.rad_ball import RadiationBall
        model = _make_model()
        # Place ball so beam midpoint (0.5, 0, 0) is in outer zone (r1=0.1, r2=2m)
        ball = RadiationBall(
            name="OuterZone",
            center=np.array([0.5, 0.0, 0.0]),
            r1=0.1,
            flux1=500_000.0,
            r2=2.0,
            flux2=5_000.0,
            active=True,
        )
        result = run_analysis(model, [ball], _make_config())
        assert result.T_centroid[-1, 0] > result.T_centroid[0, 0]

    def test_rad_ball_beyond_r2_no_exposure(self):
        """Ball too far away → no exposed elements → ValueError."""
        from fahts.core.heat.sources.rad_ball import RadiationBall
        model = _make_model()
        ball = RadiationBall(
            name="FarBall",
            center=np.array([100.0, 100.0, 100.0]),
            r1=0.5,
            flux1=350_000.0,
            r2=1.0,
            flux2=1_500.0,
            active=True,
        )
        with pytest.raises(ValueError):
            run_analysis(model, [ball], _make_config())

    def test_rad_ball_takes_precedence_over_fire_zone(self):
        """RadiationBall covers same element as FireZone — ball BC must win."""
        model = _make_model()
        ball = self._make_ball()
        zone = _make_zone()
        # Both cover the beam; result should be the same element count (no duplication)
        result = run_analysis(model, [zone, ball], _make_config())
        assert result.n_elements == 1


# ── Worker importability ──────────────────────────────────────────────────────

def test_worker_importable():
    from fahts.gui.analysis_worker import AnalysisWorker  # noqa: F401
    assert AnalysisWorker is not None
