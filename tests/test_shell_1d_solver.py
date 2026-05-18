"""Tests for Shell1DSolver — 1-D backward Euler heat solver for shell elements."""
import numpy as np
import pytest

from fahts.core.heat.section_mesh.shell_mesh import ShellMesh1D, ShellMesher
from fahts.core.heat.solver.shell_1d_solver import Shell1DSolver, _assemble_C_1d, _assemble_K_1d
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import PlateSection
from fahts.core.results.temperature_field import TemperatureField


def _make_material():
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _make_mesh(thickness=0.02, n_layers=4):
    sec = PlateSection(sid=1, thickness=thickness)
    return ShellMesher(sec, n_layers=n_layers).build()


def _const_temp(T: float):
    return lambda _t: T


class TestAssembly1D:
    def test_K_shape(self):
        mesh = _make_mesh(thickness=0.02, n_layers=4)
        K = _assemble_K_1d(mesh, k=50.0)
        n = mesh.n_nodes
        assert K.shape == (n, n)

    def test_K_row_sums_zero_at_interior(self):
        """Row sums of K_cond should be zero (uniform T → no flux)."""
        mesh = _make_mesh(n_layers=4)
        K = _assemble_K_1d(mesh, k=50.0)
        # Interior nodes
        for i in range(1, mesh.n_nodes - 1):
            assert abs(K[i, :].sum()) < 1e-10

    def test_K_symmetric(self):
        mesh = _make_mesh(n_layers=4)
        K = _assemble_K_1d(mesh, k=50.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_C_shape(self):
        mesh = _make_mesh(n_layers=4)
        C = _assemble_C_1d(mesh, rho=7850.0, cp=500.0)
        assert C.shape == (mesh.n_nodes,)

    def test_C_sum_equals_rho_cp_thickness(self):
        mesh = _make_mesh(thickness=0.02, n_layers=4)
        C = _assemble_C_1d(mesh, rho=7850.0, cp=500.0)
        expected = 7850.0 * 500.0 * 0.02
        assert abs(C.sum() - expected) < 1.0


class TestShell1DSolverStep:
    def test_step_returns_correct_shape(self):
        mesh = _make_mesh()
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(1000.0),
            epsilon_m=0.7, h_conv=25.0,
        )
        T_prev = np.full(mesh.n_nodes, 20.0)
        T_new = solver.step(T_prev, dt=30.0, t=30.0)
        assert T_new.shape == (mesh.n_nodes,)

    def test_outer_node_heats_first(self):
        """Outer face should heat up faster than inner face."""
        mesh = _make_mesh(n_layers=8)
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(800.0),
            epsilon_m=0.7, h_conv=25.0,
        )
        T_prev = np.full(mesh.n_nodes, 20.0)
        T_new = solver.step(T_prev, dt=30.0, t=30.0)
        outer = mesh.outer_node
        inner = mesh.inner_node
        assert T_new[outer] > T_new[inner], "Outer face should heat first"

    def test_cold_start_temperatures_rise(self):
        mesh = _make_mesh()
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(500.0),
            epsilon_m=0.5, h_conv=25.0,
            T0=20.0,
        )
        _, T_hist = solver.run(t_end=300.0, dt=10.0)
        assert T_hist[-1, :].mean() > 20.0

    def test_no_cooling_when_T_fire_equals_initial(self):
        """Steel at T_env should stay constant (no flux)."""
        T_env = 20.0
        mesh = _make_mesh()
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(T_env),
            epsilon_m=0.0, h_conv=25.0,
            T0=T_env,
        )
        T_prev = np.full(mesh.n_nodes, T_env)
        T_new = solver.step(T_prev, dt=60.0, t=60.0)
        np.testing.assert_allclose(T_new, T_env, atol=1e-6)


class TestShell1DSolverRun:
    def test_run_returns_correct_shapes(self):
        mesh = _make_mesh(n_layers=2)
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(500.0),
            epsilon_m=0.5, h_conv=25.0,
        )
        times, T_hist = solver.run(t_end=120.0, dt=30.0)
        # t=0 + 4 steps = 5 outputs
        assert len(times) == 5
        assert T_hist.shape == (5, mesh.n_nodes)

    def test_run_output_dt(self):
        mesh = _make_mesh(n_layers=2)
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(500.0),
            epsilon_m=0.5, h_conv=25.0,
        )
        times, T_hist = solver.run(t_end=120.0, dt=10.0, output_dt=60.0)
        # outputs at t=0, t=60, t=120 → 3 entries
        assert len(times) == 3

    def test_run_initial_temperature(self):
        mesh = _make_mesh()
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(500.0),
            epsilon_m=0.5, h_conv=25.0,
            T0=100.0,
        )
        times, T_hist = solver.run(t_end=30.0, dt=10.0)
        np.testing.assert_allclose(T_hist[0], 100.0)


class TestShell1DSolverPrescribedFlux:
    def test_q_prescribed_adds_heat(self):
        mesh = _make_mesh()
        mat = _make_material()
        solver_no_q = Shell1DSolver(
            mesh=mesh, material=mat,
            fire_temp=_const_temp(20.0), epsilon_m=0.0, h_conv=0.0,
        )
        solver_with_q = Shell1DSolver(
            mesh=mesh, material=mat,
            fire_temp=_const_temp(20.0), epsilon_m=0.0, h_conv=0.0,
            q_prescribed_fn=lambda _t: 50000.0,
        )
        T_prev = np.full(mesh.n_nodes, 20.0)
        T_no_q  = solver_no_q.step(T_prev, dt=30.0, t=30.0)
        T_with_q = solver_with_q.step(T_prev, dt=30.0, t=30.0)
        outer = mesh.outer_node
        assert T_with_q[outer] > T_no_q[outer]


class TestTemperatureFieldFromShell:
    def test_from_shell_solver_run_shape(self):
        mesh = _make_mesh(n_layers=4)
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(500.0),
            epsilon_m=0.5, h_conv=25.0,
        )
        times, T_hist = solver.run(t_end=60.0, dt=30.0)
        tf = TemperatureField.from_shell_solver_run(
            eid=42, times=times, T_history=T_hist, mesh=mesh
        )
        assert tf.element_ids == [42]
        assert tf.T_centroid.shape == (len(times), 1)
        assert 42 in tf.T_section

    def test_centroid_between_outer_and_inner(self):
        mesh = _make_mesh(n_layers=8)
        solver = Shell1DSolver(
            mesh=mesh, material=_make_material(),
            fire_temp=_const_temp(800.0),
            epsilon_m=0.7, h_conv=25.0,
        )
        times, T_hist = solver.run(t_end=300.0, dt=10.0, output_dt=60.0)
        tf = TemperatureField.from_shell_solver_run(
            eid=1, times=times, T_history=T_hist, mesh=mesh
        )
        for step_i in range(len(times)):
            T_outer = T_hist[step_i, mesh.outer_node]
            T_inner = T_hist[step_i, mesh.inner_node]
            T_cen = float(tf.T_centroid[step_i, 0])
            assert min(T_outer, T_inner) <= T_cen + 1e-6
            assert T_cen <= max(T_outer, T_inner) + 1e-6
