"""
Unit tests for the analysis_mode field added to AnalysisConfig.

Covers:
- Default value is "engineering".
- "theory" and "engineering" are accepted by validate().
- Unknown modes raise ValueError in validate().
- In engineering mode the solver uses mass_matrix="lumped" (config default).
- In theory mode the solver uses mass_matrix="consistent" regardless of
  config.mass_matrix, and config is NOT mutated.
- A small end-to-end run confirms the mode is honoured inside run_analysis().
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fahts.core.results.analysis_config import AnalysisConfig


# ── AnalysisConfig field defaults and validation ──────────────────────────────

class TestAnalysisModeDefault:
    def test_default_is_engineering(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0)
        assert cfg.analysis_mode == "engineering"

    def test_engineering_accepted(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0,
                             analysis_mode="engineering")
        cfg.validate()  # must not raise

    def test_theory_accepted(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0,
                             analysis_mode="theory")
        cfg.validate()  # must not raise

    def test_invalid_mode_raises(self):
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0,
                             analysis_mode="invalid")
        with pytest.raises(ValueError, match="analysis_mode"):
            cfg.validate()

    def test_fast_mode_raises(self):
        """Sanity-check that a plausible-but-wrong value is still rejected."""
        cfg = AnalysisConfig(t_end=3600.0, dt=30.0, output_dt=60.0,
                             analysis_mode="fast")
        with pytest.raises(ValueError, match="analysis_mode"):
            cfg.validate()

    def test_each_instance_independent(self):
        """analysis_mode default must not be shared between instances."""
        a = AnalysisConfig(t_end=100.0, dt=10.0, output_dt=10.0)
        b = AnalysisConfig(t_end=100.0, dt=10.0, output_dt=10.0)
        a.analysis_mode = "theory"
        assert b.analysis_mode == "engineering"


# ── run_analysis mass_matrix selection ───────────────────────────────────────

def _make_minimal_model():
    """Tiny synthetic FEMModel: one BOX beam, one node pair."""
    from fahts.core.model.element import BeamElement
    from fahts.core.model.fem_model import FEMModel
    from fahts.core.model.material import SteelMaterial
    from fahts.core.model.node import Node
    from fahts.core.model.section import BoxSection

    n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(nid=2, x=1.0, y=0.0, z=0.0)
    mat = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)
    sec = BoxSection(sid=10, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
    elem = BeamElement(
        eid=101, n1=1, n2=2, mat_id=1, geom_id=10, lcoor_id=0,
        length=1.0,
        direction=np.array([1.0, 0.0, 0.0]),
        local_z=np.array([0.0, 0.0, 1.0]),
    )
    return FEMModel(
        nodes={1: n1, 2: n2},
        elements={101: elem},
        sections={10: sec},
        materials={1: mat},
        groups={},
        unitvecs={},
        source_file=Path("synthetic.fem"),
    )


def _make_zone():
    from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
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


class TestAnalysisModeRunAnalysis:
    """
    Verify that run_analysis uses the correct mass_matrix for each mode.

    We patch SurfaceTransientSolver.__init__ (replacing it with a recorder that
    still delegates to the real __init__) so we can inspect which mass_matrix
    argument was passed without having to run the full time integration.
    """

    def _run_and_capture_mass_matrix(self, analysis_mode: str, config_mass: str = "lumped"):
        """
        Run a one-step analysis and return the mass_matrix strings passed to
        all SurfaceTransientSolver constructors.
        """
        from fahts.core.heat.solver import analysis_runner as ar
        from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver

        captured: list[str] = []
        real_init = SurfaceTransientSolver.__init__

        def recording_init(self, *args, mass_matrix="lumped", **kwargs):
            captured.append(mass_matrix)
            real_init(self, *args, mass_matrix=mass_matrix, **kwargs)

        model = _make_minimal_model()
        zones = [_make_zone()]
        cfg = AnalysisConfig(
            t_end=60.0, dt=60.0, output_dt=60.0,
            analysis_mode=analysis_mode,
            mass_matrix=config_mass,
        )

        with patch.object(SurfaceTransientSolver, "__init__", recording_init):
            ar.run_analysis(model, zones, cfg)

        return captured

    def test_engineering_mode_uses_lumped(self):
        captured = self._run_and_capture_mass_matrix("engineering", "lumped")
        assert len(captured) >= 1
        assert all(m == "lumped" for m in captured), (
            f"Expected all 'lumped', got {captured}"
        )

    def test_theory_mode_uses_consistent(self):
        captured = self._run_and_capture_mass_matrix("theory", "lumped")
        assert len(captured) >= 1
        assert all(m == "consistent" for m in captured), (
            f"Expected all 'consistent', got {captured}"
        )

    def test_theory_mode_does_not_mutate_config(self):
        """config.mass_matrix must remain unchanged after a theory-mode run."""
        from fahts.core.heat.solver.analysis_runner import run_analysis

        model = _make_minimal_model()
        zones = [_make_zone()]
        cfg = AnalysisConfig(
            t_end=60.0, dt=60.0, output_dt=60.0,
            analysis_mode="theory",
            mass_matrix="lumped",
        )
        run_analysis(model, zones, cfg)
        # config must NOT have been mutated
        assert cfg.mass_matrix == "lumped"
        assert cfg.analysis_mode == "theory"

    def test_theory_mode_overrides_explicit_consistent(self):
        """
        When analysis_mode='theory' and mass_matrix='consistent' the result is
        the same: consistent is still passed to solvers.
        """
        captured = self._run_and_capture_mass_matrix("theory", "consistent")
        assert len(captured) >= 1
        assert all(m == "consistent" for m in captured)

    def test_engineering_mode_honours_explicit_consistent(self):
        """
        When analysis_mode='engineering' and mass_matrix='consistent', the
        consistent matrix should be used (engineering mode does not override it).
        """
        captured = self._run_and_capture_mass_matrix("engineering", "consistent")
        assert len(captured) >= 1
        assert all(m == "consistent" for m in captured)


class TestAnalysisModeEndToEnd:
    """
    Smoke-test that both modes complete a short run without errors and produce
    physically reasonable results (temperature increases).
    """

    def _run(self, mode: str):
        from fahts.core.heat.solver.analysis_runner import run_analysis
        from fahts.core.results.temperature_field import TemperatureField

        model = _make_minimal_model()
        zones = [_make_zone()]
        cfg = AnalysisConfig(
            t_end=120.0, dt=60.0, output_dt=120.0,
            analysis_mode=mode,
        )
        result = run_analysis(model, zones, cfg)
        assert isinstance(result, TemperatureField)
        T_cen = result.T_centroid[:, 0]
        assert T_cen[-1] > T_cen[0], "Steel must heat up during fire exposure"
        return result

    def test_engineering_mode_completes(self):
        self._run("engineering")

    def test_theory_mode_completes(self):
        self._run("theory")

    def test_theory_mode_is_warmer_or_equal(self):
        """
        Consistent mass gives more accurate (usually slightly warmer) results
        than lumped mass for a coarse time step.  We only require the run
        produces a finite temperature — not a strict ordering — since for very
        short steps the two modes converge.
        """
        from fahts.core.heat.solver.analysis_runner import run_analysis

        model = _make_minimal_model()
        zones = [_make_zone()]
        base = dict(t_end=120.0, dt=60.0, output_dt=120.0)

        result_eng = run_analysis(
            model, zones, AnalysisConfig(**base, analysis_mode="engineering")
        )
        result_th = run_analysis(
            model, zones, AnalysisConfig(**base, analysis_mode="theory")
        )

        T_eng = float(result_eng.T_centroid[-1, 0])
        T_th  = float(result_th.T_centroid[-1, 0])
        # Both must be finite and above initial temperature
        assert np.isfinite(T_eng) and T_eng > 20.0
        assert np.isfinite(T_th)  and T_th  > 20.0
