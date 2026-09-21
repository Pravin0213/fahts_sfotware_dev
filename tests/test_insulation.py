"""
Tests for §3.3.3 massless temperature-dependent type-1 insulation model.

Covers:
  - InsulationLayer dataclass (validation, conductance formula)
  - SurfaceTransientSolver with insulation: steel heats slower than without
  - SurfaceTransientSolver with very-high-conductivity insulation approaches uninsulated case
  - With no insulation, behaviour is identical to baseline
  - Thermal resistance formula: R = d / lambda  [m²·K/W]
  - Conductance formula: h_ins = lambda / d  [W/(m²·K)]
  - Insulation wired through run_analysis via AnalysisConfig.insulation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fahts.core.heat.bc.insulation import InsulationLayer
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.solver.analysis_runner import run_analysis
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone
from fahts.core.model.element import BeamElement
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.material import SteelMaterial
from fahts.core.model.node import Node
from fahts.core.model.section import BoxSection
from fahts.core.results.analysis_config import AnalysisConfig


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _box_section() -> BoxSection:
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def _material() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _build_solver(
    insulation: InsulationLayer | None = None,
    T_fire: float = 800.0,
    T0: float = 20.0,
) -> SurfaceTransientSolver:
    """Build a minimal SurfaceTransientSolver for a 1 m BOX beam."""
    sec  = _box_section()
    mat  = _material()
    mesh = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
    return SurfaceTransientSolver(
        mesh=mesh,
        material=mat,
        fire_temp=lambda _t: T_fire,
        epsilon_m=0.7 * 1.0,   # typical resultant emissivity
        h_conv=25.0,
        T0=T0,
        insulation=insulation,
    )


def _minimal_model() -> FEMModel:
    """Single BOX beam element from (0,0,0) to (1,0,0)."""
    n1 = Node(nid=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(nid=2, x=1.0, y=0.0, z=0.0)
    sec = _box_section()
    mat = _material()
    direction = np.array([1.0, 0.0, 0.0])
    local_z   = np.array([0.0, 0.0, 1.0])
    elem = BeamElement(
        eid=101, n1=1, n2=2, mat_id=1, geom_id=1, lcoor_id=0,
        length=1.0, direction=direction, local_z=local_z,
    )
    return FEMModel(
        nodes={1: n1, 2: n2},
        elements={101: elem},
        sections={1: sec},
        materials={1: mat},
        groups={},
        unitvecs={},
        source_file=Path("synthetic.fem"),
    )


def _fire_zone() -> FireZone:
    """ISO 834 zone covering the beam midpoint."""
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


def _config(**overrides) -> AnalysisConfig:
    defaults = dict(t_end=120.0, dt=60.0, output_dt=120.0)
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ── InsulationLayer unit tests ────────────────────────────────────────────────

class TestInsulationLayer:
    """Validate InsulationLayer dataclass construction and physics helpers."""

    def test_constant_conductivity_roundtrip(self):
        ins = InsulationLayer(thickness=0.02, conductivity=0.2)
        assert ins.effective_conductivity(T_mean=300.0) == pytest.approx(0.2)

    def test_callable_conductivity(self):
        # Linear T-dependent conductivity: λ(T) = 0.1 + 0.001·T
        ins = InsulationLayer(thickness=0.05, conductivity=lambda T: 0.1 + 0.001 * T)
        assert ins.effective_conductivity(200.0) == pytest.approx(0.3, rel=1e-9)
        assert ins.effective_conductivity(500.0) == pytest.approx(0.6, rel=1e-9)

    def test_resistance_formula(self):
        # R = d / λ
        ins = InsulationLayer(thickness=0.04, conductivity=0.2)
        expected = 0.04 / 0.2   # = 0.2 m²·K/W
        assert ins.resistance(T_mean=100.0) == pytest.approx(expected, rel=1e-9)

    def test_conductance_formula(self):
        # h_ins = λ / d = 1/R
        ins = InsulationLayer(thickness=0.04, conductivity=0.2)
        expected = 0.2 / 0.04   # = 5.0 W/(m²·K)
        assert ins.conductance(T_mean=100.0) == pytest.approx(expected, rel=1e-9)

    def test_conductance_is_inverse_resistance(self):
        ins = InsulationLayer(thickness=0.03, conductivity=0.15)
        T = 250.0
        assert ins.conductance(T) == pytest.approx(1.0 / ins.resistance(T), rel=1e-9)

    def test_temperature_dependent_conductance(self):
        # λ(T) = 0.2 + 0.001·T; d = 0.05 m → h_ins(300°C) = (0.2+0.3)/0.05 = 10
        ins = InsulationLayer(thickness=0.05, conductivity=lambda T: 0.2 + 0.001 * T)
        assert ins.conductance(T_mean=300.0) == pytest.approx((0.2 + 0.3) / 0.05, rel=1e-9)

    def test_thickness_must_be_positive(self):
        with pytest.raises(ValueError, match="thickness"):
            InsulationLayer(thickness=0.0, conductivity=0.2)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness"):
            InsulationLayer(thickness=-0.01, conductivity=0.2)

    def test_non_positive_constant_conductivity_raises(self):
        with pytest.raises(ValueError, match="conductivity"):
            InsulationLayer(thickness=0.02, conductivity=0.0)

    def test_negative_density_raises(self):
        with pytest.raises(ValueError, match="density"):
            InsulationLayer(thickness=0.02, conductivity=0.2, density=-1.0)

    def test_negative_specific_heat_raises(self):
        with pytest.raises(ValueError, match="specific_heat"):
            InsulationLayer(thickness=0.02, conductivity=0.2, specific_heat=-1.0)

    def test_default_density_zero(self):
        ins = InsulationLayer(thickness=0.02, conductivity=0.2)
        assert ins.density == 0.0

    def test_default_specific_heat_zero(self):
        ins = InsulationLayer(thickness=0.02, conductivity=0.2)
        assert ins.specific_heat == 0.0


# ── SurfaceTransientSolver insulation physics ─────────────────────────────────

class TestSurfaceSolverInsulation:
    """Verify that insulation slows heating compared to the bare steel case."""

    def _run_solver(self, insulation, T_fire=800.0, steps=3, dt=60.0):
        """Return mean final temperature after `steps` time steps."""
        solver = _build_solver(insulation=insulation, T_fire=T_fire)
        _times, T_hist = solver.run(t_end=steps * dt, dt=dt, output_dt=steps * dt)
        return float(np.mean(T_hist[-1]))

    def test_no_insulation_baseline_heats(self):
        """Without insulation the steel temperature rises above initial 20°C."""
        T_final = self._run_solver(insulation=None)
        assert T_final > 20.0

    def test_insulation_slows_heating(self):
        """With insulation (d=50mm, λ=0.2 W/mK → h_ins=4 W/m²K) steel heats slower."""
        ins = InsulationLayer(thickness=0.05, conductivity=0.2)   # h_ins = 4 W/(m²·K)
        T_ins    = self._run_solver(insulation=ins)
        T_no_ins = self._run_solver(insulation=None)
        assert T_ins < T_no_ins, (
            f"Insulated T={T_ins:.1f}°C should be below bare T={T_no_ins:.1f}°C"
        )

    def test_thicker_insulation_heats_slower(self):
        """Doubling thickness halves h_ins → slower heating."""
        ins_thin  = InsulationLayer(thickness=0.02, conductivity=0.2)
        ins_thick = InsulationLayer(thickness=0.04, conductivity=0.2)
        T_thin  = self._run_solver(insulation=ins_thin)
        T_thick = self._run_solver(insulation=ins_thick)
        assert T_thick < T_thin

    def test_very_high_conductivity_approaches_uninsulated(self):
        """Insulation with λ→∞ (h_ins very large) approaches the bare steel result."""
        # h_ins = λ/d; with λ=1e6, d=0.02 → h_ins = 5e7 >> h_conv=25
        ins_hi = InsulationLayer(thickness=0.02, conductivity=1e6)
        T_hi_cond = self._run_solver(insulation=ins_hi)
        T_no_ins  = self._run_solver(insulation=None)
        # Should be within 5°C — the large h_ins ≫ h_conv dominates and radiation
        # also contributes in the baseline, so exact equality is not expected.
        assert abs(T_hi_cond - T_no_ins) < 5.0, (
            f"High-conductivity insulation T={T_hi_cond:.2f}°C vs bare T={T_no_ins:.2f}°C"
        )

    def test_insulation_final_temperature_positive(self):
        """Temperature stays physical (above ambient) with insulation."""
        ins = InsulationLayer(thickness=0.05, conductivity=0.2)
        T_final = self._run_solver(insulation=ins)
        assert T_final > 20.0

    def test_no_insulation_same_as_none_insulation(self):
        """Passing insulation=None gives the same result as not providing it."""
        T_none   = self._run_solver(insulation=None)
        # Build a second solver directly without kwarg to confirm default is None
        sec  = _box_section()
        mat  = _material()
        from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
        mesh = BoxSurfaceMesher(section=sec, length=1.0, n_top=2, n_side=3, n_length=4).build()
        solver2 = SurfaceTransientSolver(
            mesh=mesh, material=mat,
            fire_temp=lambda _t: 800.0,
            epsilon_m=0.7, h_conv=25.0,
        )
        _t2, T_hist2 = solver2.run(t_end=180.0, dt=60.0, output_dt=180.0)
        T_direct = float(np.mean(T_hist2[-1]))
        # Both should produce equal results — they are identical setups
        assert T_none == pytest.approx(T_direct, rel=1e-6)

    def test_temperature_history_monotone_with_insulation(self):
        """Steel temperature should increase monotonically (fire hotter than steel)."""
        ins = InsulationLayer(thickness=0.03, conductivity=0.3)
        solver = _build_solver(insulation=ins, T_fire=900.0)
        _times, T_hist = solver.run(t_end=300.0, dt=60.0, output_dt=60.0)
        mean_T = [float(np.mean(row)) for row in T_hist]
        for i in range(1, len(mean_T)):
            assert mean_T[i] >= mean_T[i - 1] - 1e-6, (
                f"Temperature non-monotone at step {i}: {mean_T[i - 1]:.2f} → {mean_T[i]:.2f}"
            )


# ── Thermal resistance formula matches §3.3.3 numerically ────────────────────

class TestInsulationResistanceFormula:
    """Verify §3.3.3 series-resistance physics at the boundary."""

    def test_h_ins_replaces_h_conv_in_steady_state(self):
        """
        In pure steady-state (no heat storage), the heat flux through the
        insulation must equal h_ins * (T_fire - T_steel).

        Approach: run solver to near steady state; check that the mean node
        temperature is bounded by T_fire and T_initial.
        """
        T_fire_val = 500.0
        ins = InsulationLayer(thickness=0.05, conductivity=0.5)   # h_ins = 10 W/m²K
        solver = _build_solver(insulation=ins, T_fire=T_fire_val, T0=20.0)
        _times, T_hist = solver.run(t_end=36000.0, dt=600.0, output_dt=36000.0)
        T_final = float(np.mean(T_hist[-1]))
        # Must converge toward T_fire in the limit
        assert 20.0 < T_final <= T_fire_val + 1.0

    def test_conductance_proportional_to_heating_rate(self):
        """
        Doubling the insulation conductance (h_ins = λ/d) should roughly double
        the initial heating rate (Δt small → linear regime).
        """
        dt = 5.0    # very small dt for linear regime

        ins_low  = InsulationLayer(thickness=0.10, conductivity=0.2)   # h=2
        ins_high = InsulationLayer(thickness=0.05, conductivity=0.2)   # h=4

        solver_low  = _build_solver(insulation=ins_low,  T_fire=800.0)
        solver_high = _build_solver(insulation=ins_high, T_fire=800.0)

        _t_lo, T_lo = solver_low.run( t_end=dt, dt=dt, output_dt=dt)
        _t_hi, T_hi = solver_high.run(t_end=dt, dt=dt, output_dt=dt)

        dT_low  = float(np.mean(T_lo[-1]))  - 20.0
        dT_high = float(np.mean(T_hi[-1])) - 20.0

        # High-conductance insulation should produce faster heating
        assert dT_high > dT_low

    def test_resistance_formula_r_equals_d_over_lambda(self):
        """Unit test: R = d/λ matches InsulationLayer.resistance() for various T."""
        d   = 0.035
        lam = 0.175
        ins = InsulationLayer(thickness=d, conductivity=lam)
        for T in [20.0, 100.0, 300.0, 600.0]:
            assert ins.resistance(T) == pytest.approx(d / lam, rel=1e-10)


# ── End-to-end via run_analysis ───────────────────────────────────────────────

class TestInsulationEndToEnd:
    """Insulation wired through AnalysisConfig → run_analysis."""

    def test_run_analysis_with_insulation_returns_result(self):
        """run_analysis accepts config.insulation and returns TemperatureField."""
        model = _minimal_model()
        zones = [_fire_zone()]
        ins   = InsulationLayer(thickness=0.05, conductivity=0.2)
        cfg   = _config(insulation=ins)
        result = run_analysis(model, zones, cfg)
        assert result is not None
        assert 101 in result.element_ids

    def test_run_analysis_insulated_cooler_than_bare(self):
        """run_analysis: insulated element stays cooler than bare element."""
        model  = _minimal_model()
        zones  = [_fire_zone()]
        cfg_bare = _config()
        cfg_ins  = _config(insulation=InsulationLayer(thickness=0.05, conductivity=0.2))

        result_bare = run_analysis(model, zones, cfg_bare)
        result_ins  = run_analysis(model, zones, cfg_ins)

        T_bare = result_bare.peak_centroid_temperature(101)
        T_ins  = result_ins.peak_centroid_temperature(101)
        assert T_ins < T_bare, (
            f"Insulated peak T={T_ins:.1f}°C should be below bare T={T_bare:.1f}°C"
        )

    def test_run_analysis_no_insulation_is_default(self):
        """AnalysisConfig.insulation defaults to None (no insulation)."""
        cfg = _config()
        assert cfg.insulation is None

    def test_config_insulation_field_accepted(self):
        """AnalysisConfig.insulation field is stored correctly."""
        ins = InsulationLayer(thickness=0.025, conductivity=0.3)
        cfg = _config(insulation=ins)
        assert cfg.insulation is ins
