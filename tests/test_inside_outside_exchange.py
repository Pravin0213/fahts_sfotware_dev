"""
Tests for §3.4.1 inside/outside energy exchange for shell profiles.

- QUADSHEL (SurfaceTransientSolver, n_exposed_sides=2): both faces heated
- TRISHELL  (Shell1DSolver, fire_temp_inner):            inner node also heated
- Hollow BOX (SurfaceTransientSolver, n_exposed_sides=1): inner adiabatic (unchanged)
"""
from __future__ import annotations

import numpy as np
import pytest

from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.shell_mesh import ShellMesher
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.shell_1d_solver import Shell1DSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import PlateSection, BoxSection


def _steel() -> SteelMaterial:
    return SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=355e6, rho=7850.0, alpha_T=1.2e-5)


def _plate_section(t: float = 0.01) -> PlateSection:
    return PlateSection(sid=1, thickness=t)


def _flat_plate_mesh(t: float = 0.01, side: float = 1.0) -> BeamSurfaceMesh:
    """1×1 m flat plate mesh via PlateSurfaceMesher."""
    corners = np.array([
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side, side, 0.0],
        [0.0, side, 0.0],
    ])
    sec = _plate_section(t)
    return PlateSurfaceMesher(section=sec, corners=corners, mesh_12=2, mesh_14=2).build()


def _fire_temp(t: float) -> float:
    return 800.0  # constant 800 °C


# ── n_exposed_sides on BeamSurfaceMesh (QUADSHEL path) ────────────────────────

class TestQuadshelDoubleSided:
    """SurfaceTransientSolver with n_exposed_sides=2 heats twice as fast as 1."""

    def _solver(self, n_sides: int) -> SurfaceTransientSolver:
        mat  = _steel()
        mesh = _flat_plate_mesh()
        return SurfaceTransientSolver(
            mesh=mesh, material=mat, fire_temp=_fire_temp,
            epsilon_m=0.5, h_conv=25.0, T0=20.0,
            n_exposed_sides=n_sides,
        )

    def test_double_sided_heats_faster(self):
        dt = 10.0
        s1 = self._solver(1)
        s2 = self._solver(2)
        mesh = _flat_plate_mesh()
        T0   = np.full(mesh.n_nodes, 20.0)

        T_one  = s1.step(T0.copy(), dt, dt)
        T_two  = s2.step(T0.copy(), dt, dt)

        assert float(np.mean(T_two)) > float(np.mean(T_one)), (
            "n_exposed_sides=2 should heat faster than n_exposed_sides=1"
        )

    def test_double_sided_approximately_double_heat_rise(self):
        """For small dt the temperature rise should be ~2× that of single-sided."""
        dt = 0.1  # small so linearity holds
        s1 = self._solver(1)
        s2 = self._solver(2)
        mesh = _flat_plate_mesh()
        T0   = np.full(mesh.n_nodes, 20.0)

        T1 = s1.step(T0.copy(), dt, dt)
        T2 = s2.step(T0.copy(), dt, dt)

        dT1 = float(np.mean(T1 - T0))
        dT2 = float(np.mean(T2 - T0))
        assert dT1 > 0.0
        ratio = dT2 / dT1
        assert abs(ratio - 2.0) < 0.05, f"Expected dT ratio ≈ 2.0, got {ratio:.4f}"

    def test_n_exposed_sides_invalid(self):
        mat  = _steel()
        mesh = _flat_plate_mesh()
        with pytest.raises(ValueError, match="n_exposed_sides"):
            SurfaceTransientSolver(
                mesh=mesh, material=mat, fire_temp=_fire_temp,
                epsilon_m=0.5, h_conv=25.0, n_exposed_sides=3,
            )


# ── Shell1DSolver inner-face fire BC (TRISHELL path) ──────────────────────────

class TestTrishellDoubleSided:
    """Shell1DSolver with fire_temp_inner heats inner node as well as outer."""

    def _solver(self, inner: bool) -> Shell1DSolver:
        sec  = _plate_section(t=0.01)
        mesh = ShellMesher(sec, n_layers=4).build()
        mat  = _steel()
        return Shell1DSolver(
            mesh=mesh, material=mat, fire_temp=_fire_temp,
            epsilon_m=0.5, h_conv=25.0, T0=20.0,
            fire_temp_inner=_fire_temp if inner else None,
        )

    def test_inner_node_heats_with_inner_bc(self):
        dt = 30.0
        s = self._solver(inner=True)
        mesh = ShellMesher(_plate_section(0.01), n_layers=4).build()
        T0 = np.full(mesh.n_nodes, 20.0)

        T = s.step(T0.copy(), dt, dt)
        assert T[mesh.inner_node] > T0[mesh.inner_node], (
            "Inner node should heat when fire_temp_inner is set"
        )

    def test_inner_heats_faster_than_outer_adiabatic(self):
        """Inner node heats faster when it has its own fire BC vs. adiabatic."""
        dt = 30.0
        s_one  = self._solver(inner=False)
        s_two  = self._solver(inner=True)
        mesh   = ShellMesher(_plate_section(0.01), n_layers=4).build()
        T0     = np.full(mesh.n_nodes, 20.0)

        T_one  = s_one.step(T0.copy(), dt, dt)
        T_two  = s_two.step(T0.copy(), dt, dt)
        inn    = mesh.inner_node

        assert T_two[inn] > T_one[inn], (
            "Inner node should be hotter with fire_temp_inner than when adiabatic"
        )

    def test_outer_node_still_heats(self):
        """Outer node (outer_node=0) should heat regardless of inner BC."""
        dt = 30.0
        for inner in (False, True):
            s    = self._solver(inner=inner)
            mesh = ShellMesher(_plate_section(0.01), n_layers=4).build()
            T0   = np.full(mesh.n_nodes, 20.0)
            T    = s.step(T0.copy(), dt, dt)
            assert T[mesh.outer_node] > T0[mesh.outer_node]

    def test_symmetric_heating_both_sides_same_temp(self):
        """When both sides see identical fire, outer and inner should converge equally."""
        dt   = 30.0  # 30 s steps; 5 mm plate equilibrates fast
        sec  = _plate_section(t=0.005)  # 5 mm plate
        mesh = ShellMesher(sec, n_layers=2).build()
        mat  = _steel()
        s    = Shell1DSolver(
            mesh=mesh, material=mat, fire_temp=_fire_temp,
            epsilon_m=0.5, h_conv=25.0, T0=20.0,
            fire_temp_inner=_fire_temp,
        )
        T = np.full(mesh.n_nodes, 20.0)
        for i in range(20):
            T = s.step(T, dt, (i + 1) * dt)

        outer_T = float(T[mesh.outer_node])
        inner_T = float(T[mesh.inner_node])
        assert np.isfinite(outer_T) and np.isfinite(inner_T)
        # Symmetric BC + thin plate → temperatures should be very close after 600 s
        assert abs(outer_T - inner_T) < 5.0, (
            f"Symmetric heating should give similar outer/inner temps; "
            f"outer={outer_T:.1f} inner={inner_T:.1f}"
        )


# ── Hollow BOX — inner adiabatic (unchanged) ──────────────────────────────────

class TestBoxInnerAdiabatic:
    """BOX hollow profile: inner surface remains adiabatic (n_exposed_sides=1)."""

    def test_single_sided_default(self):
        """SurfaceTransientSolver for BOX uses n_exposed_sides=1 by default."""
        sec  = BoxSection(sid=1, H=0.3, W=0.2, T_top=0.01, T_bot=0.01,
                          T_side=0.01)
        mesh = BoxSurfaceMesher(section=sec, length=2.0, n_top=2, n_side=2, n_length=2).build()
        mat  = _steel()
        solver = SurfaceTransientSolver(
            mesh=mesh, material=mat, fire_temp=_fire_temp,
            epsilon_m=0.5, h_conv=25.0, T0=20.0,
        )
        # n_exposed_sides defaults to 1
        assert solver._n_exposed_sides == 1

    def test_double_sided_kwarg_accepted(self):
        sec    = BoxSection(sid=1, H=0.3, W=0.2, T_top=0.01, T_bot=0.01, T_side=0.01)
        mesh   = BoxSurfaceMesher(section=sec, length=2.0, n_top=2, n_side=2, n_length=2).build()
        mat    = _steel()
        solver = SurfaceTransientSolver(
            mesh=mesh, material=mat, fire_temp=_fire_temp,
            epsilon_m=0.5, h_conv=25.0, n_exposed_sides=2,
        )
        assert solver._n_exposed_sides == 2


# ── BeamSurfaceMesh new optional fields ───────────────────────────────────────

class TestBeamSurfaceMeshNewFields:
    def test_default_empty_lists(self):
        mesh = _flat_plate_mesh()
        assert mesh.outer_face_indices == []
        assert mesh.inner_node_indices == []

    def test_box_mesh_default_empty(self):
        sec  = BoxSection(sid=1, H=0.3, W=0.2, T_top=0.01, T_bot=0.01,
                          T_side=0.01)
        mesh = BoxSurfaceMesher(section=sec, length=2.0).build()
        assert mesh.outer_face_indices == []
        assert mesh.inner_node_indices == []
