"""
Vectorised Hex8 / 3-D Quad4 finite-element kernels for the 3-D solid heat solver.

All routines are pure functions operating on a *batch* of elements at once
(leading axis n = number of elements); there are no per-element Python loops.

Hex8 conventions (VTK_HEXAHEDRON), natural coords (ξ, η, ζ) ∈ [−1, 1]³:
    0 (−,−,−)   1 (+,−,−)   2 (+,+,−)   3 (−,+,−)
    4 (−,−,+)   5 (+,−,+)   6 (+,+,+)   7 (−,+,+)
    N_a = ⅛ (1 + ξ_a ξ)(1 + η_a η)(1 + ζ_a ζ)

Quad4 (boundary face) natural coords (ξ, η) ∈ [−1, 1]², CCW node order:
    0 (−,−)   1 (+,−)   2 (+,+)   3 (−,+)

Integration: 2×2×2 Gauss for volumes, 2×2 Gauss for faces (all weights = 1).

Element base matrices (unit material properties):
    K_base = ∫ Bᵀ B dV            (conductivity with k = 1)       [m]
    C_base = ∫ Nᵀ N dV            (capacity with ρc = 1)          [m³]
    F_base = ∫ Nᵀ N dA            (boundary "face mass", h = 1)   [m²]
"""
from __future__ import annotations

import numpy as np

# ── Reference data ────────────────────────────────────────────────────────────

_G = 1.0 / np.sqrt(3.0)

#: Hex8 node signs in natural coordinates, (8, 3)
HEX8_NODE_SIGNS: np.ndarray = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)

#: Quad4 node signs in natural coordinates, (4, 2)
QUAD4_NODE_SIGNS: np.ndarray = np.array(
    [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float
)

#: 2×2×2 Gauss points (8, 3); all weights are 1
HEX8_GAUSS_POINTS: np.ndarray = _G * HEX8_NODE_SIGNS

#: 2×2 Gauss points (4, 2); all weights are 1
QUAD4_GAUSS_POINTS: np.ndarray = _G * QUAD4_NODE_SIGNS


# ── Shape functions ───────────────────────────────────────────────────────────

def hex8_shape(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Hex8 shape functions and natural derivatives at one point.

    Returns:
        N:  (8,)   shape function values.
        dN: (3, 8) derivatives w.r.t. (ξ, η, ζ).
    """
    s = HEX8_NODE_SIGNS
    a = 1.0 + s[:, 0] * xi
    b = 1.0 + s[:, 1] * eta
    c = 1.0 + s[:, 2] * zeta
    N = 0.125 * a * b * c
    dN = 0.125 * np.array([s[:, 0] * b * c, s[:, 1] * a * c, s[:, 2] * a * b])
    return N, dN


def quad4_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear Quad4 shape functions and natural derivatives at one point.

    Returns:
        N:  (4,)   shape function values.
        dN: (2, 4) derivatives w.r.t. (ξ, η).
    """
    s = QUAD4_NODE_SIGNS
    a = 1.0 + s[:, 0] * xi
    b = 1.0 + s[:, 1] * eta
    N = 0.25 * a * b
    dN = 0.25 * np.array([s[:, 0] * b, s[:, 1] * a])
    return N, dN


def _hex8_gauss_tables() -> tuple[np.ndarray, np.ndarray]:
    pairs = [hex8_shape(*gp) for gp in HEX8_GAUSS_POINTS]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _quad4_gauss_tables() -> tuple[np.ndarray, np.ndarray]:
    pairs = [quad4_shape(*gp) for gp in QUAD4_GAUSS_POINTS]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


#: Hex8 N at Gauss points (8 gp, 8 nodes) and dN (8 gp, 3, 8 nodes)
HEX8_N_GP, HEX8_DN_GP = _hex8_gauss_tables()
#: Quad4 N at Gauss points (4 gp, 4 nodes) and dN (4 gp, 2, 4 nodes)
QUAD4_N_GP, QUAD4_DN_GP = _quad4_gauss_tables()


# ── Hex8 volume kernels ───────────────────────────────────────────────────────

def hex8_jacobians(X: np.ndarray) -> np.ndarray:
    """
    Jacobian matrices J[g] = dN(g) · X at the 8 Gauss points.

    Args:
        X: (n, 8, 3) element node coordinates.

    Returns:
        (n, 8, 3, 3) Jacobians, J[e, g, a, k] = ∂x_k/∂ξ_a.
    """
    X = np.asarray(X, dtype=float)
    return np.einsum("gaj,njk->ngak", HEX8_DN_GP, X)


def hex8_gradients(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Physical shape-function gradients B and Jacobian determinants at Gauss points.

    Args:
        X: (n, 8, 3) element node coordinates.

    Returns:
        B:    (n, 8 gp, 3, 8) ∂N_j/∂x_k at each Gauss point.
        detJ: (n, 8 gp) Jacobian determinants (signed).
    """
    J = hex8_jacobians(X)
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    B = np.einsum("ngka,gaj->ngkj", invJ, HEX8_DN_GP)
    return B, detJ


def hex8_conductivity_base(X: np.ndarray) -> np.ndarray:
    """
    Element conductivity matrices for unit conductivity, K_e = ∫ Bᵀ B dV.

    Args:
        X: (n, 8, 3) element node coordinates [m].

    Returns:
        (n, 8, 8) symmetric positive semi-definite matrices [m].
    """
    B, detJ = hex8_gradients(X)
    return np.einsum("ngki,ngkj,ng->nij", B, B, detJ)


# Hex8 edges grouped by the natural direction they run along (VTK node ordering)
_HEX8_EDGES_XI = ((0, 1), (3, 2), (4, 5), (7, 6))
_HEX8_EDGES_ETA = ((0, 3), (1, 2), (4, 7), (5, 6))
_HEX8_EDGES_ZETA = ((0, 4), (1, 5), (2, 6), (3, 7))


def hex8_conductivity_twopoint_base(X: np.ndarray) -> np.ndarray:
    """
    Monotone (M-matrix) edge-based conductivity for k = 1, shape (n, 8, 8).

    Each of the 12 edges is a conductance G = k·A_dual/L, taken from the element metric
    at the centre, Gm = det J · J⁻¹J⁻ᵀ:  G_ξ = Gm₁₁/2, G_η = Gm₂₂/2, G_ζ = Gm₃₃/2 per edge.
    For a rectangular a×b×c brick this is exactly k·(bc/4)/a etc. — the 7-point finite-
    difference / two-point-flux finite-volume stencil (as in OpenFOAM or a voxel solver):
    exact for linear fields on rectangular bricks, conservative (zero row sums), and all
    off-diagonals ≤ 0, so it satisfies the discrete maximum principle whatever the aspect
    ratio.  The consistent trilinear K instead gets POSITIVE couplings between in-plane
    neighbours of thin elements (c ≪ a, b), which produces unphysical over/undershoots at
    sharp flux edges.  Non-orthogonality (skewed hexes) is neglected, as in uncorrected FV.
    """
    n = len(X)
    dN0 = hex8_shape(0.0, 0.0, 0.0)[1]                            # (3, 8)
    J = np.einsum("aj,njk->nak", dN0, X)                          # (n, 3, 3)
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    # J[a, k] = ∂x_k/∂ξ_a  →  ∂ξ_a/∂x_k = (J⁻ᵀ)[a, k]  →  metric g^{ab} = (J⁻ᵀ J⁻¹)[a, b]
    Gm = detJ[:, None, None] * np.einsum("nja,njb->nab", invJ, invJ)
    K = np.zeros((n, 8, 8))
    for d, edges in enumerate((_HEX8_EDGES_XI, _HEX8_EDGES_ETA, _HEX8_EDGES_ZETA)):
        g = 0.5 * Gm[:, d, d]
        for i, j in edges:
            K[:, i, i] += g
            K[:, j, j] += g
            K[:, i, j] -= g
            K[:, j, i] -= g
    return K


def hex8_capacity_base(X: np.ndarray, lumped: bool = True) -> np.ndarray:
    """
    Element capacity matrices for unit ρ·c.

    Args:
        X:      (n, 8, 3) element node coordinates [m].
        lumped: True → row-sum lumped diagonal (n, 8);
                False → consistent ∫ Nᵀ N dV (n, 8, 8).

    Returns:
        (n, 8) or (n, 8, 8) [m³].
    """
    detJ = np.linalg.det(hex8_jacobians(X))                    # (n, 8gp)
    if lumped:
        # Row sums of ∫NᵢNⱼ dV = ∫Nᵢ dV (partition of unity)
        return detJ @ HEX8_N_GP                                 # (n, 8)
    return np.einsum("gi,gj,ng->nij", HEX8_N_GP, HEX8_N_GP, detJ)


def hex8_volumes(X: np.ndarray) -> np.ndarray:
    """(n,) signed element volumes via 2×2×2 Gauss [m³]."""
    return np.linalg.det(hex8_jacobians(X)).sum(axis=1)


# ── Quad4 boundary-face kernels (faces embedded in 3-D) ──────────────────────

def quad3d_face_gauss(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss data for bilinear faces in 3-D space.

    Args:
        P: (n, 4, 3) face corner coordinates (CCW seen from outside).

    Returns:
        N_gp: (4 gp, 4 nodes) shape functions at the 2×2 Gauss points.
        detJ: (n, 4 gp) surface Jacobians |∂x/∂ξ × ∂x/∂η| (area scale).
    """
    P = np.asarray(P, dtype=float)
    t = np.einsum("gaj,njk->ngak", QUAD4_DN_GP, P)             # (n, 4gp, 2, 3)
    detJ = np.linalg.norm(np.cross(t[:, :, 0], t[:, :, 1]), axis=-1)
    return QUAD4_N_GP.copy(), detJ


def quad3d_face_mass_base(P: np.ndarray) -> np.ndarray:
    """
    Consistent boundary "face mass" matrices ∫ Nᵢ Nⱼ dA on 3-D bilinear faces.

    Args:
        P: (n, 4, 3) face corner coordinates.

    Returns:
        (n, 4, 4) symmetric matrices [m²]; each matrix sums to the face area.
    """
    N_gp, detJ = quad3d_face_gauss(P)
    return np.einsum("gi,gj,ng->nij", N_gp, N_gp, detJ)
