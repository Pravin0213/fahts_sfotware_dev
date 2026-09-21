"""
Tests for GlobalThermalSolver — global assembly with co-located node merging.

Verifies that the global solver produces identical results to N independent
per-element solver.step() calls when elements have no co-located nodes
(the block-diagonal case is a special case of the general DOF-map assembly).
"""
from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.solver.analysis_runner import GlobalThermalSolver
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_material() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _make_solver(fire_temp_val: float = 800.0) -> tuple:
    """Return (SurfaceTransientSolver, n_nodes) for a tiny 1 m BOX beam."""
    sec  = BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
    mat  = _make_material()
    mesh = BoxSurfaceMesher(section=sec, length=1.0, n_top=1, n_side=1, n_length=1).build()
    fire_temp = lambda t, _v=fire_temp_val: _v  # noqa: E731
    solver = SurfaceTransientSolver(
        mesh=mesh, material=mat, fire_temp=fire_temp,
        epsilon_m=0.5, h_conv=25.0, T0=20.0,
    )
    return solver, mesh.n_nodes


def _independent_gdof_map(eids: list[int], n_per_elem: dict[int, int]) -> tuple[dict, int]:
    """Build a non-overlapping DOF map (no shared nodes between elements)."""
    gdof_map: dict[int, np.ndarray] = {}
    offset = 0
    for eid in eids:
        n = n_per_elem[eid]
        gdof_map[eid] = np.arange(offset, offset + n, dtype=np.intp)
        offset += n
    return gdof_map, offset


# ── DOF layout ───────────────────────────────────────────────────────────────

class TestGlobalThermalSolverLayout:
    def test_independent_dofs_sequential(self):
        """Two independent elements get non-overlapping DOF ranges."""
        s1, n1 = _make_solver()
        s2, n2 = _make_solver(fire_temp_val=600.0)
        gdof_map, n_global = _independent_gdof_map([1, 2], {1: n1, 2: n2})
        gs = GlobalThermalSolver(
            eids=[1, 2],
            solvers={1: s1, 2: s2},
            gdof_map=gdof_map,
            n_global_dofs=n_global,
        )
        assert list(gs.gdof_map[1]) == list(range(0, n1))
        assert list(gs.gdof_map[2]) == list(range(n1, n1 + n2))
        assert gs.n_total == n1 + n2

    def test_single_element_starts_at_zero(self):
        s, n = _make_solver()
        gdof_map, n_global = _independent_gdof_map([7], {7: n})
        gs = GlobalThermalSolver(
            eids=[7],
            solvers={7: s},
            gdof_map=gdof_map,
            n_global_dofs=n_global,
        )
        assert list(gs.gdof_map[7]) == list(range(0, n))
        assert gs.n_total == n


# ── Equivalence to independent solves ────────────────────────────────────────

class TestGlobalEquivalenceToIndependent:
    """
    When elements have no co-located nodes the global solver is mathematically
    equivalent to N independent solver.step() calls.
    """

    def _run_both(self, n_steps: int = 3, dt: float = 30.0):
        """
        Run two solvers independently AND via GlobalThermalSolver.
        Returns (T_indep_1, T_indep_2, T_global_1, T_global_2) after n_steps.
        """
        s1_indep, n1 = _make_solver(fire_temp_val=800.0)
        s2_indep, n2 = _make_solver(fire_temp_val=600.0)
        s1_glob,  _  = _make_solver(fire_temp_val=800.0)
        s2_glob,  _  = _make_solver(fire_temp_val=600.0)

        T0 = np.full(n1, 20.0)

        # Independent path
        Ti_1 = T0.copy()
        Ti_2 = T0.copy()
        for step in range(1, n_steps + 1):
            t = step * dt
            Ti_1 = s1_indep.step(Ti_1, dt, t)
            Ti_2 = s2_indep.step(Ti_2, dt, t)

        # Global path — non-overlapping DOF map so elements are decoupled
        gdof_map, n_global = _independent_gdof_map([1, 2], {1: n1, 2: n2})
        gs = GlobalThermalSolver(
            eids=[1, 2],
            solvers={1: s1_glob, 2: s2_glob},
            gdof_map=gdof_map,
            n_global_dofs=n_global,
            n_workers=1,
        )
        T_glob = np.full(n_global, 20.0)
        for step in range(1, n_steps + 1):
            t = step * dt
            T_glob = gs.step(T_glob, dt, t)

        T_glob_1 = T_glob[gs.gdof_map[1]]
        T_glob_2 = T_glob[gs.gdof_map[2]]

        return Ti_1, Ti_2, T_glob_1, T_glob_2

    def test_single_step_matches(self):
        Ti_1, Ti_2, Tg_1, Tg_2 = self._run_both(n_steps=1)
        np.testing.assert_allclose(Tg_1, Ti_1, rtol=1e-6,
                                   err_msg="Element 1 mismatch after 1 step")
        np.testing.assert_allclose(Tg_2, Ti_2, rtol=1e-6,
                                   err_msg="Element 2 mismatch after 1 step")

    def test_multi_step_matches(self):
        Ti_1, Ti_2, Tg_1, Tg_2 = self._run_both(n_steps=5)
        np.testing.assert_allclose(Tg_1, Ti_1, rtol=1e-6)
        np.testing.assert_allclose(Tg_2, Ti_2, rtol=1e-6)

    def test_elements_heat_independently(self):
        """Two decoupled elements under different fire temps reach different T."""
        Ti_1, Ti_2, Tg_1, Tg_2 = self._run_both(n_steps=5)
        assert float(np.mean(Tg_1)) > float(np.mean(Tg_2))


# ── Temperature increases under fire ─────────────────────────────────────────

class TestGlobalThermalPhysics:
    def test_temperature_rises(self):
        s, n = _make_solver(fire_temp_val=800.0)
        gdof_map, n_global = _independent_gdof_map([1], {1: n})
        gs = GlobalThermalSolver(
            eids=[1], solvers={1: s}, gdof_map=gdof_map,
            n_global_dofs=n_global, n_workers=1,
        )
        T_global = np.full(n_global, 20.0)
        T_global = gs.step(T_global, 30.0, 30.0)
        assert float(np.mean(T_global)) > 20.0

    def test_no_heating_at_ambient(self):
        """Element at fire_temp=20 °C must stay near 20 °C (pure conduction, no BC flux)."""
        s, n = _make_solver(fire_temp_val=20.0)
        gdof_map, n_global = _independent_gdof_map([1], {1: n})
        gs = GlobalThermalSolver(
            eids=[1], solvers={1: s}, gdof_map=gdof_map,
            n_global_dofs=n_global, n_workers=1,
        )
        T_global = np.full(n_global, 20.0)
        T_global = gs.step(T_global, 30.0, 30.0)
        np.testing.assert_allclose(T_global, 20.0, atol=1e-6)
