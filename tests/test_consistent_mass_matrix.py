"""Tests for selectable lumped/consistent thermal mass matrices."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fahts.core.heat.section_mesh.box_mesher import BoxMesher
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.shell_mesh import ShellMesher
from fahts.core.heat.solver.fem_2d_section import (
    assemble_C_consistent,
    assemble_C_lumped,
)
from fahts.core.heat.solver.shell_1d_solver import (
    Shell1DSolver,
    _assemble_C_1d,
    _assemble_C_1d_consistent,
)
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.time_integrator import TransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig


def _material() -> SteelMaterial:
    return SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6)


def _box_section() -> BoxSection:
    return BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)


def test_analysis_config_accepts_consistent_mass_matrix():
    cfg = AnalysisConfig(
        t_end=60.0,
        dt=30.0,
        output_dt=60.0,
        mass_matrix="consistent",
    )

    cfg.validate()

    assert "mass=consistent" in cfg.summary()


def test_analysis_config_rejects_unknown_mass_matrix():
    cfg = AnalysisConfig(t_end=60.0, dt=30.0, output_dt=60.0, mass_matrix="row")

    with pytest.raises(ValueError, match="mass_matrix"):
        cfg.validate()


def test_section_consistent_mass_conserves_total_heat_capacity():
    mesh = BoxMesher(_box_section(), elem_size=0.05, n_layers=1).build()
    rho = 7850.0
    cp = 500.0

    C_lumped = assemble_C_lumped(mesh, rho, cp)
    C_consistent = assemble_C_consistent(mesh, rho, cp)

    assert sp.issparse(C_consistent)
    assert C_consistent.shape == (mesh.n_nodes, mesh.n_nodes)
    np.testing.assert_allclose(C_consistent.toarray(), C_consistent.toarray().T)
    assert C_consistent.sum() == pytest.approx(C_lumped.sum(), rel=1e-12)
    assert np.count_nonzero(C_consistent.toarray() - np.diag(C_consistent.diagonal())) > 0


def test_transient_solver_consistent_mode_assembles_sparse_mass_matrix():
    mesh = BoxMesher(_box_section(), elem_size=0.05, n_layers=1).build()
    solver = TransientSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 800.0,
        epsilon_m=0.7,
        h_conv=25.0,
        mass_matrix="consistent",
    )

    _K, M, _Q = solver._assemble_step(np.full(mesh.n_nodes, 20.0), t=0.0)

    assert sp.issparse(M)
    assert M.shape == (mesh.n_nodes, mesh.n_nodes)
    assert np.count_nonzero(M.toarray() - np.diag(M.diagonal())) > 0


def test_surface_solver_consistent_mode_assembles_sparse_mass_matrix():
    mesh = BoxSurfaceMesher(
        _box_section(), length=1.0, n_top=1, n_side=1, n_length=1
    ).build()
    solver = SurfaceTransientSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 800.0,
        epsilon_m=0.7,
        h_conv=25.0,
        mass_matrix="consistent",
    )

    _K, M, _Q = solver._assemble_step(np.full(mesh.n_nodes, 20.0), t=0.0)

    assert sp.issparse(M)
    assert M.shape == (mesh.n_nodes, mesh.n_nodes)
    assert np.count_nonzero(M.toarray() - np.diag(M.diagonal())) > 0


def test_shell_consistent_mass_conserves_total_heat_capacity():
    section = PlateSection(sid=1, thickness=0.02)
    mesh = ShellMesher(section, n_layers=4).build()
    rho = 7850.0
    cp = 500.0

    C_lumped = _assemble_C_1d(mesh, rho, cp)
    C_consistent = _assemble_C_1d_consistent(mesh, rho, cp)

    np.testing.assert_allclose(C_consistent, C_consistent.T, atol=1e-12)
    assert C_consistent.sum() == pytest.approx(C_lumped.sum(), rel=1e-12)
    assert np.count_nonzero(C_consistent - np.diag(np.diag(C_consistent))) > 0


def test_shell_solver_consistent_mode_runs_one_step():
    section = PlateSection(sid=1, thickness=0.02)
    mesh = ShellMesher(section, n_layers=2).build()
    solver = Shell1DSolver(
        mesh=mesh,
        material=_material(),
        fire_temp=lambda _t: 600.0,
        epsilon_m=0.7,
        h_conv=25.0,
        mass_matrix="consistent",
    )

    T_new = solver.step(np.full(mesh.n_nodes, 20.0), dt=30.0, t=30.0)

    assert T_new.shape == (mesh.n_nodes,)
    assert np.all(np.isfinite(T_new))
    assert solver._M_prev.ndim == 2
