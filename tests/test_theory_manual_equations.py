"""Manual-equation regression tests for the SINTEF FAHTS theory alignment list."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from fahts.core.heat.bc.net_flux import _SIGMA, radiative_flux
from fahts.core.heat.bc.view_factor import (
    geometric_view_factor,
    geometric_view_factor_double_area,
)
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.box_mesher import BoxMesher
from fahts.core.heat.section_mesh.section_mesh import SectionMesh
from fahts.core.heat.solver.analysis_runner import _compute_M_extra
from fahts.core.heat.solver.fem_2d_section import (
    add_robin_bc,
    quad4_capacity_matrix,
    quad4_conductivity_matrix,
)
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.time_integrator import TransientSolver
from fahts.core.model.material import SteelMaterial
from fahts.core.model.section import BoxSection
from fahts.core.results.temperature_field import TemperatureField


def _unit_square() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def test_eq_3_2_13_semidiscrete_balance_for_uniform_heating():
    """For uniform T and Tdot, M*Tdot + K*T equals the consistent source vector."""
    coords = _unit_square()
    rho = 2.0
    cp = 3.0
    T = np.full(4, 125.0)
    T_dot = np.full(4, 4.0)

    K = quad4_conductivity_matrix(coords, k=11.0)
    M = quad4_capacity_matrix(coords, rho=rho, cp=cp, lumped=False)

    source_density = rho * cp * T_dot[0]
    Qg = np.full(4, source_density * 1.0 / 4.0)

    np.testing.assert_allclose(M @ T_dot + K @ T, Qg, atol=1e-12)


def test_eq_3_2_15_quad4_element_matrices_match_closed_form():
    """Unit-square Quad4 K and consistent M match the manual area integrals."""
    K_expected = np.array(
        [
            [4, -1, -2, -1],
            [-1, 4, -1, -2],
            [-2, -1, 4, -1],
            [-1, -2, -1, 4],
        ],
        dtype=float,
    ) / 6.0
    M_expected = np.array(
        [
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4],
        ],
        dtype=float,
    ) / 36.0

    np.testing.assert_allclose(
        quad4_conductivity_matrix(_unit_square(), k=1.0),
        K_expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=False),
        M_expected,
        atol=1e-12,
    )


def test_eq_3_2_15_constant_boundary_flux_integrates_to_q_l_over_2_per_node():
    nodes = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    edge_pairs = np.array([[0, 1]], dtype=int)
    K = sp.lil_matrix((2, 2))
    Q = np.zeros(2)

    add_robin_bc(
        K,
        Q,
        edge_pairs,
        nodes,
        T_fire=20.0,
        T_prev=np.full(2, 20.0),
        epsilon_m=0.0,
        h_conv=0.0,
        q_prescribed=7.5,
    )

    np.testing.assert_allclose(Q, [7.5, 7.5], atol=1e-12)
    np.testing.assert_allclose(K.toarray(), 0.0, atol=1e-12)


def test_eq_3_2_30_crank_nicolson_increment_uses_current_and_previous_matrices():
    solver = TransientSolver.__new__(TransientSolver)
    solver._K_prev = sp.csr_matrix([[4.0, 1.0], [1.0, 5.0]])
    solver._M_prev = np.array([10.0, 12.0])
    solver._T_dot_prev = np.array([0.2, -0.1])
    solver._nonlinear_max_iter = 1
    solver._nonlinear_tol = 0.0

    K_i = sp.csr_matrix([[5.0, 0.5], [0.5, 6.0]])
    M_i = np.array([11.0, 13.0])
    Q_i = np.array([20.0, 30.0])
    solver._assemble_step = lambda _T, _t: (K_i, M_i, Q_i)

    T_prev = np.array([100.0, 150.0])
    dt = 2.0
    expected_A = K_i.toarray() + np.diag((2.0 / dt) * M_i)
    expected_B = Q_i - solver._K_prev @ T_prev + solver._M_prev * solver._T_dot_prev
    expected_T = T_prev + np.linalg.solve(expected_A, expected_B)

    T_new = TransientSolver.step(solver, T_prev, dt=dt, t=dt)

    np.testing.assert_allclose(T_new, expected_T, atol=1e-12)
    np.testing.assert_allclose(solver._M_prev, M_i, atol=1e-12)


def test_section_3_2_4_radiation_flux_is_negative_surface_emission():
    T_fire = 800.0
    T_steel = 200.0
    epsilon = 0.7
    steel_net_emission = epsilon * _SIGMA * (
        (T_steel + 273.15) ** 4 - (T_fire + 273.15) ** 4
    )

    assert radiative_flux(T_fire, T_steel, epsilon) == pytest.approx(
        -steel_net_emission,
        rel=1e-12,
    )


def test_section_3_2_4_view_factor_radiation_load_matches_manual_formula():
    mesh = BeamSurfaceMesh(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        quads=np.array([[0, 1, 2, 3]], dtype=int),
        thicknesses=np.array([0.01], dtype=float),
        local_cols=np.array([[0, 1]], dtype=int),
    )
    solver = SurfaceTransientSolver(
        mesh=mesh,
        material=SteelMaterial(
            mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6
        ),
        fire_temp=lambda _t: 800.0,
        epsilon_m=0.0,
        h_conv=0.0,
        epsilon_steel=0.7,
        epsilon_fire=0.6,
        view_factors=np.array([0.25]),
    )

    T_prev = np.full(mesh.n_nodes, 100.0)
    _K, _M, Q = solver._assemble_step(T_prev, t=60.0)

    expected_flux = 0.7 * _SIGMA * (
        0.25 * 0.6 * (800.0 + 273.15) ** 4 - (100.0 + 273.15) ** 4
    )
    np.testing.assert_allclose(Q, np.full(4, expected_flux / 4.0), rtol=1e-12)


def test_section_3_3_4_geometric_view_factor_matches_area_to_point_formula():
    patch_centroid = np.array([0.0, 0.0, 0.0])
    patch_normal = np.array([0.0, 0.0, 1.0])
    fire_patch = (
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        0.5,
    )

    expected = 0.5 / (math.pi * 2.0**2)

    assert geometric_view_factor(patch_centroid, patch_normal, [fire_patch]) == (
        pytest.approx(expected, rel=1e-12)
    )


# ── §3.3.4 double-area view factor tests ─────────────────────────────────────

def test_section_3_3_4_double_area_n1_equals_simplified_form():
    """With n_steel_sub=1 the double-area formula reduces to the simplified form."""
    quad = np.array([
        [-0.05, -0.05, 0.0],
        [ 0.05, -0.05, 0.0],
        [ 0.05,  0.05, 0.0],
        [-0.05,  0.05, 0.0],
    ])  # 0.1 × 0.1 m square lying in the z=0 plane
    normal = np.array([0.0, 0.0, 1.0])
    fire_patch = (np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    centroid = quad.mean(axis=0)
    expected = geometric_view_factor(centroid, normal, [fire_patch])
    result = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=1)

    assert result == pytest.approx(expected, rel=1e-12)


def test_section_3_3_4_double_area_zero_for_non_visible_face():
    """View factor is zero when normals are anti-parallel (back-facing)."""
    quad = np.array([
        [-0.05, -0.05, 0.0],
        [ 0.05, -0.05, 0.0],
        [ 0.05,  0.05, 0.0],
        [-0.05,  0.05, 0.0],
    ])
    normal = np.array([0.0, 0.0, -1.0])   # points AWAY from fire zone
    fire_patch = (np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    result = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=2)
    assert result == 0.0


def test_section_3_3_4_double_area_no_patches_returns_zero():
    quad = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    assert geometric_view_factor_double_area(quad, normal, []) == 0.0


def test_section_3_3_4_double_area_converges_with_finer_subdivision():
    """Finer steel subdivision converges toward analytical value for a large quad."""
    # Large 1×1 m steel patch at origin, small fire patch 3 m above.
    # The analytical result is the integral over the steel quad — finer subdivision
    # gives a better approximation; coarser (n=1) uses only the centroid.
    quad = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    fire_patch = (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 0.01)

    f1 = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=1)
    f4 = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=4)
    f8 = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=8)

    # All values should be positive and ≤ 1
    assert 0.0 < f1 <= 1.0
    assert 0.0 < f4 <= 1.0
    assert 0.0 < f8 <= 1.0
    # Finer subdivision should converge (|f8 - f4| < |f4 - f1|) for a
    # large patch where centroid approximation has noticeable error.
    assert abs(f8 - f4) < abs(f4 - f1)


def test_section_3_3_4_double_area_is_area_weighted_average():
    """F_12 equals the area-weighted mean of per-sub-patch simplified view factors."""
    quad = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    fire_patch = (np.array([0.5, 0.5, 4.0]), np.array([0.0, 0.0, -1.0]), 0.5)

    n = 3
    # Manually compute area-weighted average of simplified VF at each sub-patch centroid
    sub_vfs = []
    for i in range(n):
        s = (i + 0.5) / n
        for j in range(n):
            t = (j + 0.5) / n
            c = (
                (1 - s) * (1 - t) * quad[0]
                + s * (1 - t) * quad[1]
                + s * t * quad[2]
                + (1 - s) * t * quad[3]
            )
            sub_vfs.append(geometric_view_factor(c, normal, [fire_patch]))
    expected = float(np.mean(sub_vfs))  # equal-area sub-patches → simple mean

    result = geometric_view_factor_double_area(quad, normal, [fire_patch], n_steel_sub=n)
    assert result == pytest.approx(expected, rel=1e-12)


def test_section_3_3_5_heat_accumulation_mass_uses_inner_area_length_and_air_capacity():
    section = BoxSection(
        sid=1,
        H=0.20,
        W=0.10,
        T_side=0.01,
        T_bot=0.02,
        T_top=0.03,
    )
    mesh = BoxMesher(section, elem_size=0.05, n_layers=1).build()
    M_extra = _compute_M_extra(section, mesh, elem_length=3.0)

    inner_ids = mesh.inner_node_indices
    expected_total = section.inner_width * section.inner_height * 3.0 * 1200.0

    assert M_extra is not None
    assert M_extra.sum() == pytest.approx(expected_total, rel=1e-12)
    np.testing.assert_allclose(
        M_extra[inner_ids],
        expected_total / len(inner_ids),
        rtol=1e-12,
    )
    outside_ids = [i for i in range(mesh.n_nodes) if i not in set(inner_ids)]
    np.testing.assert_allclose(M_extra[outside_ids], 0.0, atol=1e-12)


def test_section_3_4_2_linearized_temperature_field_recovers_beta_y_and_beta_z():
    nodes = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    quads = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        dtype=int,
    )
    mesh = SectionMesh(
        nodes=nodes,
        quads=quads,
        outer_edge_pairs=np.empty((0, 2), dtype=int),
        inner_edge_pairs=np.empty((0, 2), dtype=int),
    )
    beta_y = -15.0
    beta_z = 35.0
    T_nodes = 120.0 + beta_y * nodes[:, 1] + beta_z * nodes[:, 0]
    field = TemperatureField(
        times=np.array([0.0]),
        element_ids=[42],
        T_centroid=np.array([[float(T_nodes.mean())]]),
        T_section={42: T_nodes[None, :]},
    )

    recovered_beta_y, recovered_beta_z = field.section_gradient(42, 0, mesh)

    assert recovered_beta_y == pytest.approx(beta_y, rel=1e-12)
    assert recovered_beta_z == pytest.approx(beta_z, rel=1e-12)


def _make_box_surface_mesh_3d(
    beam_length: float = 2.0,
    n_length: int = 2,
    half_y: float = 0.1,
    half_z: float = 0.2,
    thickness: float = 0.01,
) -> BeamSurfaceMesh:
    """
    Minimal 3-D surface mesh for a rectangular hollow box.

    Top/bottom faces: constant z = ±half_z, y in [-half_y, +half_y].
    Left/right faces: constant y = ±half_y, z in [-half_z, +half_z].
    n_length slabs along x in [0, beam_length].
    """
    xs = np.linspace(0.0, beam_length, n_length + 1)
    ys = np.array([-half_y, half_y])
    zs = np.array([-half_z, half_z])

    pool: list[np.ndarray] = []
    idx: dict[tuple, int] = {}

    def gid(x: float, y: float, z: float) -> int:
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in idx:
            idx[key] = len(pool)
            pool.append(np.array([x, y, z], dtype=float))
        return idx[key]

    quads_list: list[tuple] = []
    thick_list: list[float] = []
    lcol_list: list[tuple] = []

    for ix in range(n_length):
        # Bottom face (z = -half_z)
        quads_list.append((
            gid(xs[ix],     -half_y, -half_z),
            gid(xs[ix + 1], -half_y, -half_z),
            gid(xs[ix + 1],  half_y, -half_z),
            gid(xs[ix],      half_y, -half_z),
        ))
        thick_list.append(thickness)
        lcol_list.append((0, 1))
        # Top face (z = +half_z)
        quads_list.append((
            gid(xs[ix],      half_y,  half_z),
            gid(xs[ix + 1],  half_y,  half_z),
            gid(xs[ix + 1], -half_y,  half_z),
            gid(xs[ix],     -half_y,  half_z),
        ))
        thick_list.append(thickness)
        lcol_list.append((0, 1))
        # Left face (y = -half_y)
        quads_list.append((
            gid(xs[ix],     -half_y, -half_z),
            gid(xs[ix],     -half_y,  half_z),
            gid(xs[ix + 1], -half_y,  half_z),
            gid(xs[ix + 1], -half_y, -half_z),
        ))
        thick_list.append(thickness)
        lcol_list.append((0, 2))
        # Right face (y = +half_y)
        quads_list.append((
            gid(xs[ix],      half_y,  half_z),
            gid(xs[ix],      half_y, -half_z),
            gid(xs[ix + 1],  half_y, -half_z),
            gid(xs[ix + 1],  half_y,  half_z),
        ))
        thick_list.append(thickness)
        lcol_list.append((0, 2))

    return BeamSurfaceMesh(
        nodes=np.array(pool, dtype=float),
        quads=np.array(quads_list, dtype=np.intp),
        thicknesses=np.array(thick_list, dtype=float),
        local_cols=np.array(lcol_list, dtype=np.intp),
    )


def test_section_3_4_2_beam_surface_mesh_recovers_beta_y_and_beta_z():
    """§3.4.2 gradient via BeamSurfaceMesh: linear T(y,z) → exact (βy, βz)."""
    mesh = _make_box_surface_mesh_3d(beam_length=2.0, n_length=2, half_y=0.1, half_z=0.2)

    beta_y_true = -12.0   # gradient in z direction (bending about y-axis)
    beta_z_true =  25.0   # gradient in y direction (bending about z-axis)
    T_mean = 150.0

    nodes = mesh.nodes  # (n_nodes, 3): [x, y, z]
    T_nodes = T_mean + beta_y_true * nodes[:, 2] + beta_z_true * nodes[:, 1]

    field = TemperatureField(
        times=np.array([0.0]),
        element_ids=[1],
        T_centroid=np.array([[T_mean]]),
        T_section={1: T_nodes[None, :]},
    )

    recovered_by, recovered_bz = field.section_gradient(1, 0, mesh)

    assert recovered_by == pytest.approx(beta_y_true, rel=1e-10)
    assert recovered_bz == pytest.approx(beta_z_true, rel=1e-10)


def test_section_3_4_2_uniform_temperature_gives_zero_gradient_surface_mesh():
    """Uniform T across the surface mesh produces βy = βz = 0."""
    mesh = _make_box_surface_mesh_3d()

    T_uniform = 300.0
    T_nodes = np.full(mesh.n_nodes, T_uniform)
    field = TemperatureField(
        times=np.array([0.0]),
        element_ids=[2],
        T_centroid=np.array([[T_uniform]]),
        T_section={2: T_nodes[None, :]},
    )

    by, bz = field.section_gradient(2, 0, mesh)

    assert by == pytest.approx(0.0, abs=1e-10)
    assert bz == pytest.approx(0.0, abs=1e-10)


def test_section_3_4_2_section_gradient_dispatches_on_mesh_type():
    """section_gradient() calls the correct path for SectionMesh vs BeamSurfaceMesh."""
    # --- SectionMesh path ---
    sm_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    sm_quads = np.array([[0, 1, 2, 3]], dtype=int)
    sm = SectionMesh(
        nodes=sm_nodes,
        quads=sm_quads,
        outer_edge_pairs=np.empty((0, 2), dtype=int),
        inner_edge_pairs=np.empty((0, 2), dtype=int),
    )
    T_sm = np.array([10.0, 10.0, 30.0, 30.0])   # pure βy gradient in z
    f_sm = TemperatureField(
        times=np.array([0.0]),
        element_ids=[10],
        T_centroid=np.array([[20.0]]),
        T_section={10: T_sm[None, :]},
    )
    by_sm, bz_sm = f_sm.section_gradient(10, 0, sm)
    assert bz_sm == pytest.approx(0.0, abs=1e-10)   # no y-gradient
    assert by_sm != 0.0                              # non-zero z-gradient

    # --- BeamSurfaceMesh path: same geometry, answer must match ---
    bsm = _make_box_surface_mesh_3d(beam_length=1.0, n_length=1, half_y=1.0, half_z=1.0)
    beta_z_ref = 7.0
    T_bsm = 50.0 + beta_z_ref * bsm.nodes[:, 1]
    f_bsm = TemperatureField(
        times=np.array([0.0]),
        element_ids=[11],
        T_centroid=np.array([[50.0]]),
        T_section={11: T_bsm[None, :]},
    )
    by_bsm, bz_bsm = f_bsm.section_gradient(11, 0, bsm)
    assert bz_bsm == pytest.approx(beta_z_ref, rel=1e-10)
    assert by_bsm == pytest.approx(0.0, abs=1e-10)
