"""Regression tests for nonlinear current-step iteration in transient solvers."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fahts.core.heat.section_mesh.box_mesher import BoxMesher
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.shell_mesh import ShellMesher
from fahts.core.heat.solver.shell_1d_solver import Shell1DSolver
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.time_integrator import TransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection, PlateSection


def _material() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _box_section() -> BoxSection:
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def _exercise_picard_iteration(solver, n_nodes: int, *, sparse: bool) -> None:
    records: list[np.ndarray] = []

    def assemble_step(T_state: np.ndarray, _t: float):
        records.append(T_state.copy())
        K = sp.eye(n_nodes, format="csr") * 2.0 if sparse else np.eye(n_nodes) * 2.0
        M = np.ones(n_nodes)
        Q = np.full(n_nodes, 10.0 + 0.5 * float(np.mean(T_state)))
        return K, M, Q

    solver._assemble_step = assemble_step
    solver._K_prev = sp.eye(n_nodes, format="csr") if sparse else np.eye(n_nodes)
    solver._M_prev = np.ones(n_nodes)
    solver._T_dot_prev = np.full(n_nodes, 0.25)

    T_prev = np.zeros(n_nodes)
    T_new = solver.step(T_prev, dt=1.0, t=1.0)

    assert len(records) == solver._nonlinear_max_iter + 1
    np.testing.assert_allclose(records[0], 0.25)
    assert not np.allclose(records[1], records[0])
    np.testing.assert_allclose(records[-1], T_new)


def test_section_solver_reassembles_from_current_step_iterates():
    mesh = BoxMesher(_box_section(), elem_size=0.05, n_layers=1).build()
    solver = TransientSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 900.0,
        epsilon_m=0.7,
        h_conv=25.0,
        nonlinear_max_iter=3,
        nonlinear_tol=0.0,
    )

    _exercise_picard_iteration(solver, mesh.n_nodes, sparse=True)


def test_surface_solver_reassembles_from_current_step_iterates():
    mesh = BoxSurfaceMesher(
        _box_section(), length=1.0, n_top=1, n_side=1, n_length=1
    ).build()
    solver = SurfaceTransientSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 900.0,
        epsilon_m=0.7,
        h_conv=25.0,
        nonlinear_max_iter=3,
        nonlinear_tol=0.0,
    )

    _exercise_picard_iteration(solver, mesh.n_nodes, sparse=True)


def test_shell_solver_reassembles_from_current_step_iterates():
    section = PlateSection(sid=2, thickness=0.02)
    mesh = ShellMesher(section, n_layers=2).build()
    solver = Shell1DSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 900.0,
        epsilon_m=0.7,
        h_conv=25.0,
        nonlinear_max_iter=3,
        nonlinear_tol=0.0,
    )

    _exercise_picard_iteration(solver, mesh.n_nodes, sparse=False)
