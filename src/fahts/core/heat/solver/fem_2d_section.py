"""
2-D FEM heat transfer in steel cross-sections (Quad4 elements).

Task 3.3 — Element-level matrices
    quad4_conductivity_matrix()  K_e (4×4) via 2×2 Gauss integration
    quad4_capacity_matrix()      C_e (4×4) consistent, optionally row-sum lumped

Task 3.4 — Global assembly + Robin BC
    assemble_K()         sparse (n×n) global conductivity matrix
    assemble_C_lumped()  (n,) diagonal capacitance vector
    add_robin_bc()       in-place Robin BC: convection (semi-implicit) +
                         radiation (fully explicit using T_prev)

Coordinate frame: local y-z plane of the beam cross-section.
Node ordering convention (CCW, matching BoxMesher):
    0: bottom-left  (ξ=-1, η=-1)
    1: bottom-right (ξ=+1, η=-1)
    2: top-right    (ξ=+1, η=+1)
    3: top-left     (ξ=-1, η=+1)

Shape functions (standard Lagrange bilinear):
    N₁ = (1−ξ)(1−η)/4,  N₂ = (1+ξ)(1−η)/4
    N₃ = (1+ξ)(1+η)/4,  N₄ = (1−ξ)(1+η)/4
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from fahts.core.heat.section_mesh.section_mesh import SectionMesh

_SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]

# 2×2 Gauss quadrature rule: points in (ξ, η), all weights = 1
_G = 1.0 / np.sqrt(3.0)
_GAUSS_PTS: tuple[tuple[float, float], ...] = (
    (-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)
)


# ── Shape-function primitives ─────────────────────────────────────────────────

def _N(xi: float, eta: float) -> np.ndarray:
    """Quad4 shape functions at (ξ, η). Returns (4,)."""
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def _dN_nat(xi: float, eta: float) -> np.ndarray:
    """
    Quad4 shape-function natural derivatives. Returns (2, 4):
        row 0: ∂N/∂ξ
        row 1: ∂N/∂η
    """
    return 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi)],
    ])


def _jacobian(xi: float, eta: float, coords: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute Jacobian J = ∂(y,z)/∂(ξ,η) and its determinant.

    Args:
        xi, eta: natural coordinates
        coords:  (4, 2) node [y, z] coordinates

    Returns:
        (J, det_J) where J is (2, 2)
    """
    dN = _dN_nat(xi, eta)           # (2, 4)
    J = dN @ coords                  # (2, 4) @ (4, 2) → (2, 2)
    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, det_J


def _B_matrix(xi: float, eta: float, coords: np.ndarray) -> np.ndarray:
    """
    Physical gradient matrix B = J⁻¹ · ∂N/∂(ξ,η). Returns (2, 4):
        row 0: ∂N/∂y
        row 1: ∂N/∂z
    """
    dN = _dN_nat(xi, eta)
    J, det_J = _jacobian(xi, eta, coords)
    J_inv = np.array([[ J[1, 1], -J[0, 1]],
                      [-J[1, 0],  J[0, 0]]]) / det_J
    return J_inv @ dN                # (2, 2) @ (2, 4) → (2, 4)


# ── Element matrices ──────────────────────────────────────────────────────────

def quad4_conductivity_matrix(coords: np.ndarray, k: float) -> np.ndarray:
    """
    Compute the 4×4 Quad4 element conductivity (stiffness) matrix.

    K_e[i,j] = ∫∫ k (∂Nᵢ/∂y ∂Nⱼ/∂y + ∂Nᵢ/∂z ∂Nⱼ/∂z) dA

    Integration: 2×2 Gauss quadrature (exact for axis-aligned rectangles,
    consistent for general parallelogram-shaped quads).

    Args:
        coords: (4, 2) node [y, z] coordinates in CCW order.
        k:      thermal conductivity [W/(m·K)].

    Returns:
        (4, 4) symmetric positive semi-definite matrix.
        Row sums are zero (uniform temperature → zero flux).
    """
    K = np.zeros((4, 4))
    for xi, eta in _GAUSS_PTS:
        B = _B_matrix(xi, eta, coords)              # (2, 4)
        _, det_J = _jacobian(xi, eta, coords)
        K += k * (B.T @ B) * det_J                 # weight = 1.0
    return K


def quad4_capacity_matrix(
    coords: np.ndarray,
    rho: float,
    cp: float,
    lumped: bool = True,
) -> np.ndarray:
    """
    Compute the 4×4 Quad4 element capacitance (mass) matrix.

    Consistent form:
        C_e[i,j] = ∫∫ ρ·cₚ · Nᵢ · Nⱼ dA

    Row-sum lumped form (default, per ROADMAP — avoids oscillations):
        C_lump = diag(row_sums(C_consistent))
        Each node receives 1/4 of the element's thermal mass (for rectangles).

    Args:
        coords: (4, 2) node [y, z] coordinates in CCW order.
        rho:    density [kg/m³].
        cp:     specific heat [J/(kg·K)].
        lumped: if True, apply row-sum lumping (returns diagonal matrix).

    Returns:
        (4, 4) symmetric positive definite matrix (consistent or lumped diagonal).
    """
    C = np.zeros((4, 4))
    for xi, eta in _GAUSS_PTS:
        N = _N(xi, eta)                             # (4,)
        _, det_J = _jacobian(xi, eta, coords)
        C += rho * cp * np.outer(N, N) * det_J     # weight = 1.0
    if lumped:
        C = np.diag(C.sum(axis=1))
    return C


# ── Global assembly ───────────────────────────────────────────────────────────

def assemble_K(mesh: SectionMesh, k: float) -> sp.csr_matrix:
    """
    Assemble the global conductivity matrix from all Quad4 elements.

    Uses COO triplet accumulation (duplicate entries are summed by scipy).

    Args:
        mesh: SectionMesh from BoxMesher.
        k:    thermal conductivity [W/(m·K)], treated as uniform.

    Returns:
        (n_nodes, n_nodes) sparse CSR matrix, symmetric positive semi-definite.
        Row sums are zero before any BC is applied.
    """
    n = mesh.n_nodes
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for quad in mesh.quads:
        K_e = quad4_conductivity_matrix(mesh.nodes[quad], k)
        for li in range(4):
            for lj in range(4):
                rows.append(int(quad[li]))
                cols.append(int(quad[lj]))
                vals.append(K_e[li, lj])

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def assemble_C_lumped(mesh: SectionMesh, rho: float, cp: float) -> np.ndarray:
    """
    Assemble the global lumped capacitance vector (diagonal of C_global).

    Each node accumulates ρ·cₚ · (element area / 4) from each attached element.

    Args:
        mesh: SectionMesh from BoxMesher.
        rho:  density [kg/m³].
        cp:   specific heat [J/(kg·K)].

    Returns:
        (n_nodes,) array — the diagonal of the lumped mass matrix [J/(m·K)].
        Sum equals ρ·cₚ·A_steel (mass conservation per unit beam length).
    """
    C = np.zeros(mesh.n_nodes)
    for quad in mesh.quads:
        C_e = quad4_capacity_matrix(mesh.nodes[quad], rho, cp, lumped=True)
        for li in range(4):
            C[int(quad[li])] += C_e[li, li]
    return C


def assemble_C_consistent(mesh: SectionMesh, rho: float, cp: float) -> sp.csr_matrix:
    """
    Assemble the global consistent capacitance matrix from Quad4 elements.

    Args:
        mesh: SectionMesh from a 2-D section mesher.
        rho:  density [kg/m³].
        cp:   specific heat [J/(kg·K)].

    Returns:
        (n_nodes, n_nodes) sparse CSR mass matrix [J/(m·K)].
        Sum of all entries equals ρ·cₚ·A_steel.
    """
    n = mesh.n_nodes
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for quad in mesh.quads:
        C_e = quad4_capacity_matrix(mesh.nodes[quad], rho, cp, lumped=False)
        for li in range(4):
            for lj in range(4):
                rows.append(int(quad[li]))
                cols.append(int(quad[lj]))
                vals.append(C_e[li, lj])

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ── Robin boundary condition ──────────────────────────────────────────────────

def add_robin_bc(
    K: sp.lil_matrix,
    f: np.ndarray,
    edge_pairs: np.ndarray,
    nodes: np.ndarray,
    T_fire: float,
    T_prev: np.ndarray,
    epsilon_m: float,
    h_conv: float,
    *,
    q_prescribed: float = 0.0,
) -> None:
    """
    Add Robin (convective + radiative) BC contributions to K and f in-place.

    For each exposed boundary edge (node pair a–b of length L):

    Convection — semi-implicit (adds to K to improve stability):
        K_conv = h_conv · L/6 · [[2, 1], [1, 2]]
        f_conv = h_conv · T_fire · L/2 · [1, 1]

    Radiation — fully explicit using T_prev (nonlinear; EN 1993-1-2 §3.1):
        q_rad(x) = ε_m · σ · (T_fire_K⁴ − T_prev_K(x)⁴)
        Integrated with linear shape functions along the edge:
            f_rad[a] = L/6 · (2·q_rad_a + q_rad_b)
            f_rad[b] = L/6 · (q_rad_a + 2·q_rad_b)

    Prescribed flux (USERFLUX / RadiationBall) — added directly to load vector:
        f_presc[a] = f_presc[b] = q_prescribed · L/2
        Set epsilon_m=0 and h_conv to ball's h_conv when using this option.

    Net flux is zero when T_prev = T_fire everywhere (equilibrium consistency).

    Args:
        K:            (n, n) lil_matrix, modified in-place.
        f:            (n,) load vector, modified in-place.
        edge_pairs:   (n_edges, 2) exposed boundary edge node-index pairs.
        nodes:        (n_nodes, 2) node [y, z] coordinates [m].
        T_fire:       fire/gas temperature [°C].
        T_prev:       nodal temperatures at previous time step [°C], shape (n,).
        epsilon_m:    resultant emissivity (fire × steel surface), typically 0.5–0.8.
                      Set to 0.0 when using q_prescribed for RadiationBall.
        h_conv:       convective coefficient [W/(m²·K)].
        q_prescribed: constant prescribed flux [W/m²] added to load vector only.
                      Used by RadiationBall where radiation is prescribed directly.
    """
    T_fire_K = T_fire + 273.15

    for a, b in edge_pairs:
        a, b = int(a), int(b)
        L = np.hypot(nodes[b, 0] - nodes[a, 0], nodes[b, 1] - nodes[a, 1])

        # ── Convective BC (semi-implicit) ──────────────────────────────────
        h_L6 = h_conv * L / 6.0
        K[a, a] += 2.0 * h_L6
        K[a, b] += h_L6
        K[b, a] += h_L6
        K[b, b] += 2.0 * h_L6

        h_Tfire_L2 = h_conv * T_fire * L / 2.0
        f[a] += h_Tfire_L2
        f[b] += h_Tfire_L2

        # ── Radiative flux (fully explicit, EN 1993-1-2 §3.1) ──────────────
        T_a_K = T_prev[a] + 273.15
        T_b_K = T_prev[b] + 273.15
        q_a = epsilon_m * _SIGMA * (T_fire_K**4 - T_a_K**4)
        q_b = epsilon_m * _SIGMA * (T_fire_K**4 - T_b_K**4)

        # Shape-function-consistent integration along linear edge
        L6 = L / 6.0
        f[a] += L6 * (2.0 * q_a + q_b)
        f[b] += L6 * (q_a + 2.0 * q_b)

        # ── Prescribed flux (RadiationBall / USERFLUX) ──────────────────────
        if q_prescribed:
            f[a] += q_prescribed * L * 0.5
            f[b] += q_prescribed * L * 0.5
