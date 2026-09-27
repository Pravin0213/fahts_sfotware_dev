"""Tests for the vectorised Hex8 / 3-D Quad4 FEM kernels (fem_3d.py)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fahts.core.heat.solid_mesh.solid_mesh import SolidMesh
from fahts.core.heat.solver.fem_3d import (
    HEX8_GAUSS_POINTS,
    hex8_capacity_base,
    hex8_conductivity_base,
    hex8_shape,
    hex8_volumes,
    quad3d_face_gauss,
    quad3d_face_mass_base,
    quad4_shape,
)

UNIT_CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
)

# VTK hex local faces ordered CCW seen from outside
_HEX_FACES = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
)


def block_mesh(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int,
               jitter: float = 0.0, seed: int = 0) -> SolidMesh:
    """Structured block Hex8 mesh; interior nodes optionally jittered."""
    xs, ys, zs = (np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1),
                  np.linspace(0, lz, nz + 1))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    if jitter:
        rng = np.random.default_rng(seed)
        h = np.array([lx / nx, ly / ny, lz / nz])
        interior = ((nodes > 1e-12) & (nodes < np.array([lx, ly, lz]) - 1e-12)).all(axis=1)
        nodes[interior] += jitter * h * rng.uniform(-1, 1, (interior.sum(), 3))

    def nid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    K, J, I = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    hexes = np.stack([nid(I, J, K), nid(I + 1, J, K), nid(I + 1, J + 1, K), nid(I, J + 1, K),
                      nid(I, J, K + 1), nid(I + 1, J, K + 1), nid(I + 1, J + 1, K + 1),
                      nid(I, J + 1, K + 1)], axis=1)
    all_f = hexes[:, _HEX_FACES].reshape(-1, 4)
    key = np.sort(all_f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    faces = all_f[cnt[inv.ravel()] == 1]
    return SolidMesh(nodes=nodes, hexes=hexes, faces=faces,
                     face_group=np.zeros(len(faces), dtype=int))


def assemble(mesh: SolidMesh) -> np.ndarray:
    Ke = hex8_conductivity_base(mesh.nodes[mesh.hexes])
    n = mesh.n_nodes
    K = np.zeros((n, n))
    for e, h in enumerate(mesh.hexes):
        K[np.ix_(h, h)] += Ke[e]
    return K


# ── Shape functions ───────────────────────────────────────────────────────────

def test_hex8_shape_partition_of_unity_and_kronecker():
    for p in HEX8_GAUSS_POINTS:
        N, dN = hex8_shape(*p)
        assert N.sum() == pytest.approx(1.0)
        assert np.allclose(dN.sum(axis=1), 0.0)
    signs = 2 * UNIT_CUBE - 1
    for a, s in enumerate(signs):
        N, _ = hex8_shape(*s)
        assert np.allclose(N, np.eye(8)[a])


def test_quad4_shape_partition_of_unity():
    N, dN = quad4_shape(0.3, -0.7)
    assert N.sum() == pytest.approx(1.0)
    assert np.allclose(dN.sum(axis=1), 0.0)


# ── Conductivity ──────────────────────────────────────────────────────────────

def test_unit_cube_K_row_sums_zero_symmetric_psd():
    K = hex8_conductivity_base(UNIT_CUBE[None])[0]
    assert np.allclose(K.sum(axis=1), 0.0, atol=1e-14)
    assert np.allclose(K, K.T)
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-12
    assert np.sum(np.abs(w) < 1e-10) == 1          # only the constant mode
    assert np.allclose(np.diag(K), 1.0 / 3.0)       # classic Hex8 value (h = 1)


def test_K_scales_linearly_with_size():
    K1 = hex8_conductivity_base(UNIT_CUBE[None])[0]
    K2 = hex8_conductivity_base(2.5 * UNIT_CUBE[None])[0]
    assert np.allclose(K2, 2.5 * K1)


def test_rotation_translation_invariance_distorted():
    rng = np.random.default_rng(3)
    X = UNIT_CUBE + 0.15 * rng.uniform(-1, 1, (8, 3))
    R = Rotation.from_euler("xyz", [0.4, -1.1, 2.3]).as_matrix()
    Xr = X @ R.T + np.array([350.0, -12.0, 4.0])
    Xs = np.stack([X, Xr])
    K = hex8_conductivity_base(Xs)
    C = hex8_capacity_base(Xs, lumped=False)
    assert np.allclose(K[0], K[1], atol=1e-10)
    assert np.allclose(C[0], C[1], atol=1e-12)
    assert np.allclose(K[0].sum(axis=1), 0.0, atol=1e-12)


def test_patch_test_linear_field_exact_boundary_flux():
    """K·T(linear) equals ∮ Nᵢ (∇T·n) dA on the boundary; zero at interior nodes."""
    mesh = block_mesh(1.0, 0.6, 0.4, 3, 3, 3, jitter=0.25, seed=1)
    grad = np.array([2.0, -3.0, 5.0])
    T = mesh.nodes @ grad + 7.0
    KT = assemble(mesh) @ T

    P = mesh.nodes[mesh.faces]
    rows = quad3d_face_mass_base(P).sum(axis=2)          # ∫Nᵢ dA
    flux = mesh.face_normals() @ grad                    # k∇T·n (k = 1)
    f = np.zeros(mesh.n_nodes)
    np.add.at(f, mesh.faces.ravel(), (rows * flux[:, None]).ravel())
    assert np.allclose(KT, f, atol=1e-11)
    boundary = np.zeros(mesh.n_nodes, bool)
    boundary[mesh.faces.ravel()] = True
    assert np.allclose(KT[~boundary], 0.0, atol=1e-11)


# ── Capacity ──────────────────────────────────────────────────────────────────

def test_capacity_sums_equal_volume():
    mesh = block_mesh(1.0, 0.5, 0.2, 2, 3, 2, jitter=0.2)
    X = mesh.nodes[mesh.hexes]
    V = mesh.volume
    assert V == pytest.approx(0.1)
    Cl = hex8_capacity_base(X, lumped=True)
    Cc = hex8_capacity_base(X, lumped=False)
    assert Cl.shape == (mesh.n_hexes, 8)
    assert Cc.shape == (mesh.n_hexes, 8, 8)
    assert Cl.sum() == pytest.approx(V)
    assert Cc.sum() == pytest.approx(V)
    assert np.allclose(Cc.sum(axis=2), Cl)
    assert np.allclose(hex8_volumes(X), mesh.hex_volumes())
    rho_c = 7850 * 600
    assert (rho_c * Cl).sum() == pytest.approx(rho_c * V)


def test_consistent_capacity_unit_cube_values():
    C = hex8_capacity_base(UNIT_CUBE[None], lumped=False)[0]
    assert C[0, 0] == pytest.approx(8 / 216)           # 1/27
    assert C[0, 6] == pytest.approx(1 / 216)           # opposite corner


# ── Faces ─────────────────────────────────────────────────────────────────────

def test_face_mass_sums_to_area_planar_and_curved():
    P_flat = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], float)
    P_curved = np.array([[0, 0, 0], [1, 0, 0.2], [1, 1, 0], [0, 1, 0.3]], float)
    P = np.stack([P_flat, P_curved])
    F = quad3d_face_mass_base(P)
    assert F[0].sum() == pytest.approx(6.0)
    mesh = SolidMesh(nodes=P.reshape(-1, 3), hexes=np.zeros((0, 8), int),
                     faces=np.arange(8).reshape(2, 4), face_group=np.zeros(2, int))
    assert np.allclose(F.sum(axis=(1, 2)), mesh.face_areas())
    assert np.allclose(F, F.transpose(0, 2, 1))
    assert F[0, 0, 0] == pytest.approx(6.0 / 9.0)     # A/9 on the diagonal


def test_face_gauss_detJ_integrates_area_and_rotation_invariant():
    P = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], float)
    R = Rotation.from_euler("zyx", [0.3, 0.9, -0.2]).as_matrix()
    Ps = np.stack([P, P @ R.T + 5.0])
    N_gp, detJ = quad3d_face_gauss(Ps)
    assert N_gp.shape == (4, 4) and detJ.shape == (2, 4)
    assert np.allclose(detJ.sum(axis=1), 6.0)
    assert np.allclose(N_gp.sum(axis=1), 1.0)
