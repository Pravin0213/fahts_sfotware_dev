"""Tests for SolidTransientSolver (3-D Hex8 transient heat solver)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fahts.core.heat.bc.inner_robin_bc import InnerRobinBC
from fahts.core.heat.bc.insulation import InsulationLayer
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.solid_mesh.solid_mesh import FACE_INNER, FACE_OUTER, SolidMesh
from fahts.core.heat.solver.solid_solver import SolidTransientSolver, material_properties
from fahts.core.model.material import SteelMaterial

_SIGMA = 5.67e-8
_HEX_FACES = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
)


@dataclass
class ConstMat:
    k: float = 50.0
    c: float = 500.0
    rho: float = 7850.0

    def conductivity(self, T: float) -> float:  # noqa: ARG002
        return self.k

    def specific_heat(self, T: float) -> float:  # noqa: ARG002
        return self.c


def block_mesh(lx, ly, lz, nx, ny, nz, inner_z0: bool = False) -> SolidMesh:
    """Structured block; all faces FACE_OUTER (or the z = 0 face FACE_INNER)."""
    xs, ys, zs = (np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1),
                  np.linspace(0, lz, nz + 1))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    K, J, I = (a.ravel() for a in np.meshgrid(np.arange(nz), np.arange(ny),
                                              np.arange(nx), indexing="ij"))
    hexes = np.stack([nid(I, J, K), nid(I + 1, J, K), nid(I + 1, J + 1, K), nid(I, J + 1, K),
                      nid(I, J, K + 1), nid(I + 1, J, K + 1), nid(I + 1, J + 1, K + 1),
                      nid(I, J + 1, K + 1)], axis=1)
    all_f = hexes[:, _HEX_FACES].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(all_f, axis=1), axis=0,
                            return_inverse=True, return_counts=True)
    faces = all_f[cnt[inv.ravel()] == 1]
    group = np.full(len(faces), FACE_OUTER)
    if inner_z0:
        group[np.all(nodes[faces][:, :, 2] < 1e-12, axis=1)] = FACE_INNER
    mesh = SolidMesh(nodes=nodes, hexes=hexes, faces=faces, face_group=group)
    mesh.validate()
    return mesh


def _const(v: float):
    return lambda _t: v


def _energy(solver: SolidTransientSolver, T: np.ndarray, T0: float) -> float:
    _, M, _ = solver._assemble_step(T, 0.0)
    return float(np.sum(M * (T - T0)))


# ── Basic behaviour ───────────────────────────────────────────────────────────

def test_adiabatic_no_temperature_change():
    mesh = block_mesh(0.2, 0.1, 0.02, 3, 2, 2)
    mat = SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=3.55e8, rho=7850, alpha_T=1.2e-5)
    s = SolidTransientSolver(mesh, mat, _const(900.0), epsilon_m=0.0, h_conv=0.0,
                             epsilon_steel=0.0)
    _, T = s.run(t_end=300.0, dt=10.0)
    assert np.allclose(T, 20.0, atol=1e-10)


def test_lumped_capacitance_convection_small_biot():
    """Thin, conductive plate under convection → T = Tf + (T0−Tf)·exp(−hAt/ρcV)."""
    lx, ly, lz = 0.3, 0.2, 0.005
    mesh = block_mesh(lx, ly, lz, 3, 2, 2)
    mat = ConstMat()
    h, Tf, T0 = 25.0, 500.0, 20.0
    s = SolidTransientSolver(mesh, mat, _const(Tf), epsilon_m=0.0, h_conv=h, T0=T0)
    times, T = s.run(t_end=1800.0, dt=5.0, output_dt=300.0)
    A = 2 * (lx * ly + lx * lz + ly * lz)
    V = lx * ly * lz
    exact = Tf + (T0 - Tf) * np.exp(-h * A * times / (mat.rho * mat.c * V))
    assert h * lz / mat.k < 0.01
    assert np.allclose(T @ mesh.node_volume_weights, exact, rtol=2e-3)
    assert np.ptp(T[-1]) < 0.5                         # nearly uniform (Bi << 1)


def test_radiation_matches_lumped_ode():
    lx, ly, lz = 0.3, 0.2, 0.005
    mesh = block_mesh(lx, ly, lz, 2, 2, 1)
    mat = ConstMat()
    eps, Tf = 0.56, 900.0
    s = SolidTransientSolver(mesh, mat, _const(Tf), epsilon_m=eps, h_conv=0.0)
    times, T = s.run(t_end=900.0, dt=2.0, output_dt=300.0)
    A = 2 * (lx * ly + lx * lz + ly * lz)
    V = lx * ly * lz

    def rhs(_t, y):
        return [eps * _SIGMA * A * ((Tf + 273.15) ** 4 - (y[0] + 273.15) ** 4)
                / (mat.rho * mat.c * V)]

    ref = solve_ivp(rhs, (0, 900), [20.0], t_eval=times, rtol=1e-10, atol=1e-10)
    assert np.allclose(T @ mesh.node_volume_weights, ref.y[0], rtol=2e-3)


def test_view_factor_unity_equals_emissivity_mode():
    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    n_o = len(mesh.outer_face_indices)
    a = SolidTransientSolver(mesh, ConstMat(), _const(800.0), epsilon_m=0.6, h_conv=25.0)
    b = SolidTransientSolver(mesh, ConstMat(), _const(800.0), epsilon_m=0.6, h_conv=25.0,
                             epsilon_steel=0.6, epsilon_fire=1.0,
                             view_factors=np.ones(n_o))
    assert np.allclose(a.run(600.0, 10.0)[1], b.run(600.0, 10.0)[1])


def test_zero_exposure_no_heating():
    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    n_o = len(mesh.outer_face_indices)
    s = SolidTransientSolver(mesh, ConstMat(), _const(800.0), epsilon_m=0.6, h_conv=25.0,
                             face_exposure=np.zeros(n_o))
    _, T = s.run(600.0, 10.0)
    assert np.allclose(T, 20.0)


def test_per_face_array_shape_validated():
    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    with pytest.raises(ValueError):
        SolidTransientSolver(mesh, ConstMat(), _const(800.0), 0.6, 25.0,
                             q_per_face=np.ones(3))


# ── Energy balance ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mass_matrix", ["lumped", "consistent"])
def test_energy_balance_prescribed_uniform_flux(mass_matrix):
    lx, ly, lz = 0.2, 0.1, 0.03
    mesh = block_mesh(lx, ly, lz, 4, 2, 3)
    q, t_end = 2.0e4, 600.0
    s = SolidTransientSolver(mesh, ConstMat(), _const(0.0), epsilon_m=0.0, h_conv=0.0,
                             q_prescribed_fn=_const(q), epsilon_steel=0.0,
                             mass_matrix=mass_matrix)
    _, T = s.run(t_end, 10.0)
    A = 2 * (lx * ly + lx * lz + ly * lz)
    if mass_matrix == "lumped":
        E = _energy(s, T[-1], 20.0)
    else:
        _, M, _ = s._assemble_step(T[-1], 0.0)
        E = float(np.ones(mesh.n_nodes) @ (M @ (T[-1] - 20.0)))
    assert E == pytest.approx(q * A * t_end, rel=1e-9)


def test_energy_balance_directional_flux_one_face():
    """q_per_face on the z = lz face only → energy = q·A_top·t; gradient through z."""
    lx, ly, lz = 0.2, 0.2, 0.05
    mesh = block_mesh(lx, ly, lz, 2, 2, 5)
    cent = mesh.face_centroids()[mesh.outer_face_indices]
    q_face = np.where(cent[:, 2] > lz - 1e-9, 5.0e4, 0.0)
    s = SolidTransientSolver(mesh, ConstMat(), _const(0.0), epsilon_m=0.0, h_conv=0.0,
                             q_per_face=q_face, epsilon_steel=0.0)
    _, T = s.run(300.0, 5.0)
    assert _energy(s, T[-1], 20.0) == pytest.approx(5.0e4 * lx * ly * 300.0, rel=1e-9)
    top = mesh.nodes[:, 2] > lz - 1e-9
    bot = mesh.nodes[:, 2] < 1e-9
    assert T[-1][top].mean() > T[-1][bot].mean() + 1.0


def test_reradiation_reduces_temperature():
    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    kw = dict(fire_temp=_const(0.0), epsilon_m=0.0, h_conv=0.0,
              q_prescribed_fn=_const(5.0e4))
    a = SolidTransientSolver(mesh, ConstMat(), epsilon_steel=0.0, **kw).run(600.0, 10.0)[1]
    b = SolidTransientSolver(mesh, ConstMat(), epsilon_steel=0.7, **kw).run(600.0, 10.0)[1]
    assert b[-1].mean() < a[-1].mean() - 1.0


# ── Boundary conditions ───────────────────────────────────────────────────────

def test_prescribed_node_dirichlet_respected():
    mesh = block_mesh(0.5, 0.05, 0.05, 10, 1, 1)
    left = np.flatnonzero(mesh.nodes[:, 0] < 1e-9).tolist()
    bc = PrescribedNodeBC(node_indices=left, temperature=500.0)
    s = SolidTransientSolver(mesh, ConstMat(), _const(20.0), epsilon_m=0.0, h_conv=0.0,
                             epsilon_steel=0.0, prescribed_node_bcs=[bc])
    _, T = s.run(600.0, 10.0)
    assert np.allclose(T[1:, left], 500.0)
    right = mesh.nodes[:, 0] > 0.5 - 1e-9
    assert 20.0 < T[-1][right].mean() < T[-1][~right].max()
    x = mesh.nodes[:, 0]
    T_last = T[-1]
    order = np.argsort(x)
    assert np.all(np.diff(T_last[order][::4]) <= 1e-9)   # monotone decreasing in x


def test_prescribed_node_index_out_of_range():
    mesh = block_mesh(0.1, 0.1, 0.1, 1, 1, 1)
    with pytest.raises(ValueError):
        SolidTransientSolver(mesh, ConstMat(), _const(20.0), 0.0, 0.0,
                             prescribed_node_bcs=[PrescribedNodeBC([99], 100.0)])


def test_inner_robin_bc_exchanges_heat():
    lx, ly, lz = 0.1, 0.1, 0.004
    mesh = block_mesh(lx, ly, lz, 2, 2, 1, inner_z0=True)
    assert len(mesh.inner_face_indices) == 4
    h_in, Tfl = 200.0, 150.0
    base = dict(fire_temp=_const(20.0), epsilon_m=0.0, h_conv=0.0, epsilon_steel=0.0)
    T_adi = SolidTransientSolver(mesh, ConstMat(), **base).run(600.0, 5.0)[1]
    assert np.allclose(T_adi, 20.0)

    s = SolidTransientSolver(mesh, ConstMat(), inner_bc=InnerRobinBC(h_in, Tfl), **base)
    times, T = s.run(600.0, 5.0, output_dt=100.0)
    mat = ConstMat()
    exact = Tfl + (20.0 - Tfl) * np.exp(-h_in * lx * ly * times / (mat.rho * mat.c
                                                                     * lx * ly * lz))
    assert np.allclose(T.mean(axis=1), exact, rtol=5e-3)


def test_inner_robin_callable_fluid_cooling():
    mesh = block_mesh(0.1, 0.1, 0.01, 1, 1, 2, inner_z0=True)
    bc = InnerRobinBC(h=lambda t: 50.0, T_fluid=lambda t: 0.0)
    s = SolidTransientSolver(mesh, ConstMat(), _const(20.0), 0.0, 0.0, T0=100.0,
                             epsilon_steel=0.0, inner_bc=bc)
    _, T = s.run(300.0, 10.0)
    assert np.all(np.diff(T.mean(axis=1)) < 0.0)
    bottom = mesh.nodes[:, 2] < 1e-9
    assert T[-1][bottom].mean() < T[-1][~bottom].mean()


def test_insulation_equals_convection_with_h_ins():
    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    ins = InsulationLayer(thickness=0.02, conductivity=0.1)
    a = SolidTransientSolver(mesh, ConstMat(), _const(900.0), epsilon_m=0.7, h_conv=25.0,
                             insulation=ins).run(600.0, 10.0)[1]
    b = SolidTransientSolver(mesh, ConstMat(), _const(900.0), epsilon_m=0.0, h_conv=5.0,
                             ).run(600.0, 10.0)[1]
    assert np.allclose(a, b)


# ── Time integration ──────────────────────────────────────────────────────────

def test_crank_nicolson_second_order_in_dt():
    mesh = block_mesh(0.2, 0.05, 0.05, 4, 1, 1)
    left = np.flatnonzero(mesh.nodes[:, 0] < 1e-9).tolist()
    kw = dict(fire_temp=_const(20.0), epsilon_m=0.0, h_conv=0.0, epsilon_steel=0.0)

    def run(dt):
        bc = PrescribedNodeBC(left, lambda t: 20.0 + 0.5 * t)
        mat = ConstMat(k=50.0)
        return SolidTransientSolver(mesh, mat, prescribed_node_bcs=[bc], **kw
                                    ).run(400.0, dt)[1][-1]

    ref = run(0.5)
    e1 = np.max(np.abs(run(40.0) - ref))
    e2 = np.max(np.abs(run(20.0) - ref))
    e3 = np.max(np.abs(run(10.0) - ref))
    assert e1 / e2 == pytest.approx(4.0, rel=0.3)
    assert e2 / e3 == pytest.approx(4.0, rel=0.3)


def test_nonlinear_steel_heating_finite_monotone():
    mesh = block_mesh(0.3, 0.1, 0.02, 3, 2, 2)
    mat = SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=3.55e8, rho=7850, alpha_T=1.2e-5)
    fire = lambda t: 20.0 + 345.0 * np.log10(8.0 * t / 60.0 + 1.0)  # noqa: E731
    s = SolidTransientSolver(mesh, mat, fire, epsilon_m=0.7, h_conv=25.0)
    _, T = s.run(1800.0, 15.0, output_dt=150.0)
    assert np.all(np.isfinite(T))
    assert np.all(np.diff(T.mean(axis=1)) > 0.0)
    assert T[-1].max() < fire(1800.0)


# ── Material properties & driver compatibility ───────────────────────────────

@pytest.mark.parametrize("usfos", [False, True])
def test_material_properties_vectorised_match_scalar(usfos):
    mat = SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=3.55e8, rho=7850, alpha_T=1.2e-5,
                        usfos_mode=usfos)
    T = np.array([-10.0, 20.0, 123.456, 599.9, 700.0, 734.0, 750.0, 800.0, 1100.0, 1250.0])
    k, c = material_properties(mat, T)
    k_ref = np.array([mat.conductivity(x) for x in T])
    c_ref = np.array([mat.specific_heat(x) for x in T])
    assert np.allclose(k, k_ref, rtol=1e-4)
    assert np.allclose(c, c_ref, rtol=2e-3)


def test_per_hex_properties_used():
    """A hot hex gets its own (higher) EC3 specific heat → larger nodal capacity."""
    mesh = block_mesh(0.2, 0.1, 0.1, 2, 1, 1)
    mat = SteelMaterial(mid=1, E=2.1e11, nu=0.3, fy=3.55e8, rho=7850, alpha_T=1.2e-5)
    s = SolidTransientSolver(mesh, mat, _const(20.0), 0.0, 0.0, epsilon_steel=0.0)
    _, M, _ = s._assemble_step(np.where(mesh.nodes[:, 0] > 0.15, 700.0, 20.0), 0.0)
    far_right = mesh.nodes[:, 0] > 0.2 - 1e-9
    far_left = mesh.nodes[:, 0] < 1e-9
    assert M[far_right].mean() > M[far_left].mean()     # c(700) > c(20)


def test_drivable_by_global_thermal_solver():
    from fahts.core.heat.solver.analysis_runner import GlobalThermalSolver

    mesh = block_mesh(0.1, 0.1, 0.01, 2, 2, 1)
    mk = lambda: SolidTransientSolver(mesh, ConstMat(), _const(800.0),  # noqa: E731
                                      epsilon_m=0.6, h_conv=25.0)
    local = mk()
    g = GlobalThermalSolver([1], {1: mk()}, {1: np.arange(mesh.n_nodes)}, mesh.n_nodes)
    T_l = np.full(mesh.n_nodes, 20.0)
    T_g = T_l.copy()
    local._init_rate(T_l)
    for i in range(1, 6):
        T_l = local.step(T_l, 10.0, 10.0 * i)
        T_g = g.step(T_g, 10.0, 10.0 * i)
    assert np.allclose(T_l, T_g, atol=1e-8)
