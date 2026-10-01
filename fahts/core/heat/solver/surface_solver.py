"""
Crank-Nicolson transient heat solver for a 3-D beam surface mesh.

Governing equation (2-D in face plane, axial × hoop):
    M · Ṫ + K · T = Q(T)

Element matrices for a surface element with face area A_e and wall thickness t:
    K_e = t × quad4_conductivity_matrix(coords_2d, k)   [W/K]
    M_e = t × quad4_capacity_matrix(coords_2d, ρ, cp)   [J/K]

Fire boundary condition — shape-function-consistent integration (Eq. 3.2.15):
    K_conv[i,j] = h_conv × ∫∫ Nᵢ·Nⱼ dA   (consistent; combined into K_e in one COO pass)
    Q_conv[i]   = h_conv × T_fire × ∫∫ Nᵢ dA
    Q_rad[i]    = ∫∫ Nᵢ × ε·σ·(T_fire_K⁴ − T(ξ,η)_K⁴) dA   (Gauss-integrated)
    Q_presc[i]  = q_prescribed × ∫∫ Nᵢ dA

Crank-Nicolson (θ=1/2) incremental form (SINTEF FAHTS Eq. 3.2.30):
    A · ΔTi = B
    A = Ki + (2/Δt) · Mi
    B = Qi − K_{i−1}·T_{i−1} + M_{i−1}·Ṫ_{i−1}
    Ti = T_{i−1} + ΔTi
    Ṫi = (2/Δt)·ΔTi − Ṫ_{i−1}

Optional heat accumulation element:
    M_extra[node] += enclosed-fluid heat capacity [J/K]
    Used for hollow BOX/PIPE members per SINTEF FAHTS §3.3.5.

Nonlinear terms are handled by bounded Picard iteration within each time step:
K_i, M_i, and Q_i are reassembled from the current temperature iterate until
the current-step temperature converges.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fahts.core.heat.bc.insulation import InsulationLayer
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.model.material import SteelMaterial
from fahts.core.heat.solver.fem_2d_section import (
    quad4_conductivity_matrix,
    quad4_capacity_matrix,
)

# Suppress SciPy spsolve efficiency warning for small dense-ish systems
import warnings as _warnings

_SIGMA = 5.67e-8   # Stefan-Boltzmann constant [W/(m²·K⁴)]

# 2×2 Gauss rule (for element area computation)
_G = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))

MassMatrix = np.ndarray | sp.csr_matrix


def _mass_to_matrix(M: MassMatrix, scale: float = 1.0) -> sp.csr_matrix:
    """Return a sparse matrix representation of vector-lumped or full mass."""
    if sp.issparse(M):
        return M.tocsr() * scale
    return sp.diags(scale * M)


def _mass_matvec(M: MassMatrix, v: np.ndarray) -> np.ndarray:
    """Multiply vector-lumped or full mass by a vector."""
    return M @ v if sp.issparse(M) else M * v


def _mass_solve(M: MassMatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve M*x = rhs for vector-lumped or full mass."""
    if sp.issparse(M):
        return spla.spsolve(M.tocsr(), rhs)
    return rhs / M


# ── Element-level helpers ─────────────────────────────────────────────────────

def _quad_area(coords_2d: np.ndarray) -> float:
    """
    Face area of a Quad4 element via 2×2 Gauss integration.

    Args:
        coords_2d: (4, 2) local node coordinates in face plane [m].

    Returns:
        Element face area [m²].
    """
    area = 0.0
    for xi, eta in _GAUSS_PTS:
        dN = 0.25 * np.array([
            [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi)],
        ])
        J = dN @ coords_2d
        area += J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]  # weight = 1
    return abs(area)


# ── Global assembly ───────────────────────────────────────────────────────────

def assemble_surface_K(mesh: BeamSurfaceMesh, k: float) -> sp.csr_matrix:
    """
    Global conductivity matrix for a beam surface mesh.

    K_e = thickness × quad4_conductivity_matrix(local_2d_coords, k)

    Returns:
        (n_nodes, n_nodes) sparse CSR matrix.
    """
    n = mesh.n_nodes
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for q, quad in enumerate(mesh.quads):
        t = float(mesh.thicknesses[q])
        coords_2d = mesh.element_coords_2d(q)
        K_e = t * quad4_conductivity_matrix(coords_2d, k)
        for li in range(4):
            for lj in range(4):
                rows.append(int(quad[li]))
                cols.append(int(quad[lj]))
                vals.append(K_e[li, lj])

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def assemble_surface_M(
    mesh: BeamSurfaceMesh,
    rho: float,
    cp: float,
    *,
    lumped: bool = True,
) -> MassMatrix:
    """
    Global capacitance for a beam surface mesh.

    M_e = thickness × quad4_capacity_matrix(local_2d_coords, ρ, cp)

    Returns:
        (n_nodes,) lumped diagonal or (n_nodes,n_nodes) consistent matrix [J/K].
    """
    if not lumped:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for q, quad in enumerate(mesh.quads):
            t = float(mesh.thicknesses[q])
            coords_2d = mesh.element_coords_2d(q)
            C_e = t * quad4_capacity_matrix(coords_2d, rho, cp, lumped=False)
            for li in range(4):
                for lj in range(4):
                    rows.append(int(quad[li]))
                    cols.append(int(quad[lj]))
                    vals.append(C_e[li, lj])
        return sp.csr_matrix((vals, (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes))

    M = np.zeros(mesh.n_nodes)
    for q, quad in enumerate(mesh.quads):
        t = float(mesh.thicknesses[q])
        coords_2d = mesh.element_coords_2d(q)
        C_e = t * quad4_capacity_matrix(coords_2d, rho, cp, lumped=True)
        for li in range(4):
            M[int(quad[li])] += C_e[li, li]
    return M


def add_surface_fire_bc(
    K_lil: sp.lil_matrix,
    Q: np.ndarray,
    mesh: BeamSurfaceMesh,
    T_fire: float,
    T_prev: np.ndarray,
    epsilon_m: float,
    h_conv: float,
    q_prescribed: float = 0.0,
) -> None:
    """
    Add fire BC contributions to K and Q for all surface elements (in-place).

    Shape-function-consistent integration over each element's face area (Eq. 3.2.15):
        K_conv[i,j] = h_conv × ∫∫ Nᵢ·Nⱼ dA  (full 4×4, not diagonal)
        Q_conv[i]   = h_conv × T_fire × ∫∫ Nᵢ dA
        Q_rad[i]    = ∫∫ Nᵢ × ε_m·σ·(T_fire_K⁴ − T(ξ,η)_K⁴) dA  (Gauss-integrated)
        Q_presc[i]  = q_prescribed × ∫∫ Nᵢ dA
    """
    T_fire_K = T_fire + 273.15
    N_gauss = np.array([
        0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ])
        for xi, eta in _GAUSS_PTS
    ])  # (4, 4) [gp, node]

    for q, quad in enumerate(mesh.quads):
        coords_2d = mesh.element_coords_2d(q)
        M_e = quad4_capacity_matrix(coords_2d, 1.0, 1.0, lumped=False)  # ∫∫ Ni·Nj dA

        det_J_gp = np.zeros(4)
        for gp_idx, (xi, eta) in enumerate(_GAUSS_PTS):
            dN = 0.25 * np.array([
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
            ])
            J = dN @ coords_2d
            det_J_gp[gp_idx] = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])

        T_nodes_K = T_prev[quad] + 273.15                # (4,)
        T_gp_K = N_gauss @ T_nodes_K                    # (4,) [gp]
        q_rad_gp = epsilon_m * _SIGMA * (T_fire_K ** 4 - T_gp_K ** 4)
        f_rad = N_gauss.T @ (q_rad_gp * det_J_gp)       # (4,) [node]

        for li in range(4):
            a = int(quad[li])
            M_row = M_e[li]
            for lj in range(4):
                K_lil[a, int(quad[lj])] += h_conv * M_row[lj]
            Q[a] += h_conv * T_fire * float(M_row.sum())
            Q[a] += float(f_rad[li])
            if q_prescribed:
                Q[a] += q_prescribed * float(M_row.sum())


# ── Solver ────────────────────────────────────────────────────────────────────

class SurfaceTransientSolver:
    """
    Crank-Nicolson (θ=1/2) 2-D FEM heat solver for a beam surface mesh.

    Solves the heat equation in the axial × hoop plane of each shell element.
    Wall thickness enters as a parameter in the element matrices, not as a
    mesh dimension (SINTEF FAHTS §3.2.2 approach).

    Interface is identical to TransientSolver so analysis_runner can
    dispatch to either solver transparently.

    Args:
        mesh:            BeamSurfaceMesh produced by BoxSurfaceMesher.
        material:        SteelMaterial — provides k(T), cp(T), rho.
        fire_temp:       Callable(t [s]) → T_fire [°C].
        epsilon_m:       Resultant emissivity (fire × steel surface).
        h_conv:          Convective coefficient [W/(m²·K)].
        T0:              Initial uniform temperature [°C] (default 20.0).
        q_prescribed_fn: Optional Callable(t) → prescribed flux [W/m²].
        q_per_quad:      Optional (n_quads,) static per-quad flux [W/m²].
                         Used for RadiationBall falloff (§3.5.4) where each
                         face receives a different flux based on cos(θ) to ball.
        quad_exposure:   Optional (n_quads,) exposure fractions in [0, 1].
                         When set, each quad's fire-zone BC (convection, radiation,
                         prescribed flux) is scaled by its exposure value.  Allows
                         partial exposure for beams straddling a FireZone boundary.
                         Does NOT affect q_per_quad (RadiationBall) or re-radiation.
        M_extra:         Optional extra lumped capacitance [J/K per node] for
                         hollow-profile heat accumulation (§3.3.5).
        mass_matrix:     "lumped" (default) or "consistent".
        nonlinear_max_iter: Maximum Picard iterations per time step.
        nonlinear_tol:   Relative max-norm convergence tolerance for Picard iteration.
        n_exposed_sides: 1 (default) or 2.  Set to 2 for QUADSHEL/TRISHELL elements
                         where both face surfaces are exposed to fire (§3.4.1).
                         Scales convection, radiation, and prescribed-flux BC terms by 2.
                         Per-quad directional flux (q_per_quad) is NOT scaled.
        insulation:      Optional InsulationLayer (§3.3.3).  When set, the direct
                         convective and radiative fire BC terms are suppressed and
                         replaced by a linearised insulation conductance h_ins =
                         λ_i(T_mean) / d_i in both the stiffness matrix and load
                         vector.  T_mean = (T_fire + mean(T_nodes)) / 2 is recomputed
                         at each Picard iterate so temperature-dependent λ is handled
                         consistently.
        prescribed_node_bcs: Optional list of PrescribedNodeBC (§3.5.2).  Each entry
                         pins the listed surface-mesh node indices to the given
                         temperature (constant or callable) at every time step via
                         Dirichlet row-zeroing / diagonal-pinning elimination applied
                         to the assembled system before the linear solve.  Prescribed
                         DOFs are excluded from the CN update: after solving, T_new[i]
                         is overwritten with the prescribed value and T_dot[i] is set
                         to zero so the rate term does not drift.
    """

    def __init__(
        self,
        mesh: BeamSurfaceMesh,
        material: SteelMaterial,
        fire_temp: Callable[[float], float],
        epsilon_m: float,
        h_conv: float,
        T0: float = 20.0,
        q_prescribed_fn: Callable[[float], float] | None = None,
        epsilon_steel: float | None = None,
        epsilon_fire: float | None = None,
        view_factors: np.ndarray | None = None,
        q_per_quad: np.ndarray | None = None,
        quad_exposure: np.ndarray | None = None,
        M_extra: np.ndarray | None = None,
        mass_matrix: str = "lumped",
        nonlinear_max_iter: int = 6,
        nonlinear_tol: float = 1e-6,
        n_exposed_sides: int = 1,
        insulation: InsulationLayer | None = None,
        prescribed_node_bcs: list[PrescribedNodeBC] | None = None,
    ) -> None:
        if nonlinear_max_iter < 1:
            raise ValueError("nonlinear_max_iter must be >= 1")
        if nonlinear_tol < 0.0:
            raise ValueError("nonlinear_tol must be >= 0")
        if mass_matrix not in {"lumped", "consistent"}:
            raise ValueError("mass_matrix must be 'lumped' or 'consistent'")
        if n_exposed_sides not in {1, 2}:
            raise ValueError("n_exposed_sides must be 1 or 2")

        self._mesh      = mesh
        self._mat       = material
        self._fire_temp = fire_temp
        self._eps       = epsilon_m
        self._h_conv    = h_conv
        self._T0        = T0
        self._q_fn: Callable[[float], float] = (
            q_prescribed_fn if q_prescribed_fn is not None else lambda _t: 0.0
        )

        # §3.2.4 radiation with §3.3.4 geometric view factors
        if view_factors is not None:
            self._use_vf    = True
            self._eps_steel = epsilon_steel if epsilon_steel is not None else epsilon_m
            self._eps_fire  = epsilon_fire  if epsilon_fire  is not None else 1.0
            self._F_arr     = np.asarray(view_factors, dtype=float)  # (n_quads,)
        else:
            self._use_vf    = False
            self._eps_steel = epsilon_m
            self._eps_fire  = 1.0
            self._F_arr     = None

        # §3.5.4 per-face falloff flux for RadiationBall beyond r2 (time-invariant)
        # q_per_quad may be a static array or a callable t → (n_quads,) (time-varying
        # point/line-source power — see fahts/core/heat/bc/face_flux.py)
        self._q_quad_fn: Callable[[float], np.ndarray] | None = (
            q_per_quad if callable(q_per_quad) else None
        )
        if self._q_quad_fn is not None:
            q_per_quad = self._q_quad_fn(0.0)
        self._q_per_quad: np.ndarray | None = (
            np.asarray(q_per_quad, dtype=float) if q_per_quad is not None else None
        )
        # Heat-transfer-element-level exposure fractions (0.0–1.0 per quad)
        self._quad_exposure: np.ndarray | None = (
            np.asarray(quad_exposure, dtype=float) if quad_exposure is not None else None
        )
        self._M_extra: np.ndarray | None = (
            np.asarray(M_extra, dtype=float) if M_extra is not None else None
        )
        if self._M_extra is not None and self._M_extra.shape != (mesh.n_nodes,):
            raise ValueError(
                f"M_extra must have shape ({mesh.n_nodes},), got {self._M_extra.shape}"
            )
        self._mass_matrix = mass_matrix
        self._nonlinear_max_iter = int(nonlinear_max_iter)
        self._nonlinear_tol = float(nonlinear_tol)
        # §3.4.1 shell/plate elements: both faces exposed; =1 for beams
        self._n_exposed_sides = int(n_exposed_sides)

        # §3.2.4 steel surface re-radiation for prescribed-flux (RadiationBall) mode.
        # When epsilon_m > 0 (FireZone), q_rad = ε_m·σ·(T_fire⁴ − T_steel⁴) already
        # contains the -T_steel⁴ term.  When epsilon_m = 0 (prescribed flux), there is
        # no fire temperature and q_rad = 0, so re-radiation must be applied separately:
        #   Q_rerad[node] = −ε_steel · σ · T_node_K⁴ · A_e/4
        self._eps_rerad: float = (
            (epsilon_steel if epsilon_steel is not None else 0.7)
            if epsilon_m == 0.0 else 0.0
        )

        # §3.3.3 insulation layer (type 1 — massless thermal resistance)
        self._insulation: InsulationLayer | None = insulation

        # §3.5.2 prescribed nodal boundary temperatures (Dirichlet BCs)
        self._prescribed_bcs: list[PrescribedNodeBC] = (
            list(prescribed_node_bcs) if prescribed_node_bcs else []
        )
        # Validate node indices against mesh size
        for bc in self._prescribed_bcs:
            for idx in bc.node_indices:
                if idx < 0 or idx >= mesh.n_nodes:
                    raise ValueError(
                        f"PrescribedNodeBC node index {idx} is out of range for "
                        f"mesh with {mesh.n_nodes} nodes."
                    )

        # CN state — initialised by _init_rate()
        self._K_prev:     sp.csr_matrix | None = None
        self._M_prev:     MassMatrix | None    = None
        self._T_dot_prev: np.ndarray | None    = None

        # Pre-computed geometry cache (time-invariant)
        self._precompute_geometry()

    # ── Geometry pre-computation ──────────────────────────────────────────────

    def _precompute_geometry(self) -> None:
        """Cache all time-invariant geometry so per-step assembly is vectorised."""
        mesh = self._mesh
        n_q  = mesh.n_quads
        n    = mesh.n_nodes

        # 2-D coords for every element: (n_quads, 4, 2)
        all_coords = np.array([mesh.element_coords_2d(q) for q in range(n_q)])

        # Conductivity base matrices K_e / k: (n_quads, 4, 4)
        self._K_base: np.ndarray = np.array([
            quad4_conductivity_matrix(all_coords[q], 1.0) for q in range(n_q)
        ])

        # Lumped capacity base diagonals / (rho*cp): (n_quads, 4)
        self._M_base: np.ndarray = np.array([
            quad4_capacity_matrix(all_coords[q], 1.0, 1.0, lumped=True).diagonal()
            for q in range(n_q)
        ])
        self._M_consistent_base: np.ndarray = np.array([
            quad4_capacity_matrix(all_coords[q], 1.0, 1.0, lumped=False)
            for q in range(n_q)
        ])

        # COO row/col index arrays for sparse K assembly: (n_quads*16,)
        # rows[q,li,lj] = quads[q,li]; cols[q,li,lj] = quads[q,lj]
        q_arr = mesh.quads.astype(np.intp)
        self._rows_K: np.ndarray = np.repeat(q_arr, 4, axis=1).ravel()
        self._cols_K: np.ndarray = np.tile(q_arr, (1, 4)).ravel()

        # Element areas / 4 for BC: (n_quads,)
        self._areas_4: np.ndarray = (
            np.array([_quad_area(all_coords[q]) for q in range(n_q)]) * 0.25
        )

        # Wall thicknesses: (n_quads,)
        self._t_arr: np.ndarray = np.asarray(mesh.thicknesses, dtype=float)

        # Per-quad fire-BC exposure scale (0.0–1.0); default all-ones (fully exposed)
        if self._quad_exposure is not None:
            if len(self._quad_exposure) != n_q:
                raise ValueError(
                    f"quad_exposure must have {n_q} elements, got {len(self._quad_exposure)}"
                )
            exp = self._quad_exposure
        else:
            exp = np.ones(n_q)
        self._exp_1d: np.ndarray = exp               # (n_quads,)
        self._exp_2d: np.ndarray = exp[:, None]      # (n_quads, 1) — broadcast with (n_quads, 4)
        self._exp_3d: np.ndarray = exp[:, None, None]  # (n_quads, 1, 1) — with (n_quads, 4, 4)

        # Flat quad-node indices for np.add.at BC assembly: (n_quads*4,)
        self._quads_flat: np.ndarray = mesh.quads.ravel().astype(np.intp)
        self._n = n

        # ── Consistent BC geometry (Eq. 3.2.15) ──────────────────────────────
        # Shape functions at 4 Gauss points: (n_gp=4, n_nodes_per_elem=4)
        self._N_gauss: np.ndarray = np.array([
            0.25 * np.array([
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ])
            for xi, eta in _GAUSS_PTS
        ])  # (4, 4) [gp, local_node]

        # Jacobian determinants at each Gauss point: (n_quads, 4)
        det_J = np.zeros((n_q, 4))
        for q_idx in range(n_q):
            c = all_coords[q_idx]
            for gp_idx, (xi, eta) in enumerate(_GAUSS_PTS):
                dN = 0.25 * np.array([
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
                ])
                J = dN @ c
                det_J[q_idx, gp_idx] = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
        self._det_J_gauss: np.ndarray = det_J  # (n_quads, 4) [quad, gp]

        # Row sums of M_consistent_base: ∫∫ Nᵢ dA per node per element (n_quads, 4)
        self._M_consistent_rowsum: np.ndarray = self._M_consistent_base.sum(axis=2)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assemble_step(
        self, T_prev: np.ndarray, t: float
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Vectorised assembly using pre-cached geometry.

        K is built from conductivity + consistent convective BC stiffness
        ∫∫ h·Nᵢ·Nⱼ dA in a single COO → CSR pass.  Radiation and flux loads
        are integrated at 2×2 Gauss points (Eq. 3.2.15) so the BC matches
        the Galerkin consistency of the element conductivity matrix.
        """
        T_mean   = float(np.mean(T_prev))
        k        = self._mat.conductivity(T_mean)
        cp       = self._mat.specific_heat(T_mean)
        rho      = self._mat.rho
        T_fire   = self._fire_temp(t)
        T_fire_K = T_fire + 273.15
        q_presc  = self._q_fn(t)
        n        = self._n

        # ── Insulation BC stiffness coefficient (§3.3.3 type-1 massless) ─────────
        # When insulation is present, the standard convective/radiative fire BC is
        # replaced by a linearised series-resistance conductance:
        #   h_ins = λ_i(T_mean) / d_i   [W/(m²·K)]
        # where T_mean = (T_fire + mean(T_steel)) / 2 — the FAHTS characteristic
        # temperature used to evaluate the temperature-dependent conductivity.
        # h_ins replaces h_conv in the K stiffness and load vector; the direct
        # radiation/convection terms from the fire are suppressed.
        if self._insulation is not None:
            T_mean_ins = 0.5 * (T_fire + T_mean)
            h_ins = self._insulation.conductance(T_mean_ins)
        else:
            h_ins = self._h_conv  # standard convective coefficient

        # ── Conductivity + consistent convective/insulation BC stiffness ──────
        # K_e = t·k·K_base[q] + n_sides·h_bc·exposure[q]·M_consistent_base[q]
        # h_bc is h_conv (no insulation) or h_ins (insulated).
        K_vals = (
            k * self._t_arr[:, None, None] * self._K_base
            + self._n_exposed_sides * h_ins * self._exp_3d * self._M_consistent_base
        ).ravel()
        K = sp.csr_matrix((K_vals, (self._rows_K, self._cols_K)), shape=(n, n))

        # ── Capacity matrix ───────────────────────────────────────────────────
        if self._mass_matrix == "consistent":
            M_vals = (
                rho * cp * self._t_arr[:, None, None] * self._M_consistent_base
            ).ravel()
            M_i: MassMatrix = sp.csr_matrix(
                (M_vals, (self._rows_K, self._cols_K)), shape=(n, n)
            )
            if self._M_extra is not None:
                M_i = M_i + sp.diags(self._M_extra)
        else:
            M_vals = (rho * cp * self._t_arr[:, None] * self._M_base).ravel()
            M_i = np.zeros(n)
            np.add.at(M_i, self._quads_flat, M_vals)
            if self._M_extra is not None:
                M_i += self._M_extra

        # ── Fire BC load vector ───────────────────────────────────────────────
        Q_i = np.zeros(n)

        if self._insulation is not None:
            # §3.3.3 insulation path: replace conv+rad BC with insulation leakage.
            # The energy leakage E_l = h_ins·(T_fire − T_steel) is applied as a
            # linearised Robin BC:  K += h_ins·∫∫NiNj dA (done above in K_vals),
            #                       Q += h_ins·T_fire·∫∫Ni dA
            # T_fire acts as the fixed outer boundary temperature (outer insulation
            # surface temperature is coupled to T_fire via the massless assumption
            # i.e. it adjusts instantly to balance radiation+convection from the fire).
            np.add.at(Q_i, self._quads_flat,
                      (self._n_exposed_sides * h_ins * T_fire
                       * self._exp_2d * self._M_consistent_rowsum).ravel())
        else:
            # Standard (un-insulated) fire BC ────────────────────────────────────
            # Convection load: n_sides·h_conv·exposure[q] × T_fire × ∫∫ Nᵢ dA
            np.add.at(Q_i, self._quads_flat,
                      (self._n_exposed_sides * self._h_conv * T_fire
                       * self._exp_2d * self._M_consistent_rowsum).ravel())

            # Temperature at Gauss points — used by radiation and re-radiation
            T_nodes_K = T_prev[self._mesh.quads] + 273.15    # (n_quads, 4)
            T_gp_K    = T_nodes_K @ self._N_gauss.T          # (n_quads, 4) [quad, gp]

            # Radiation — §3.2.4/§3.3.4, Gauss-point integrated (Eq. 3.2.15)
            if self._use_vf:
                q_rad_gp = self._eps_steel * _SIGMA * (
                    self._F_arr[:, None] * self._eps_fire * T_fire_K ** 4
                    - T_gp_K ** 4
                )
            else:
                q_rad_gp = self._eps * _SIGMA * (T_fire_K ** 4 - T_gp_K ** 4)
            # f_rad[i] = n_sides·exposure[q] · Σ_gp N_i(gp) × q_rad(gp) × |J(gp)|
            f_rad_elem = (q_rad_gp * self._det_J_gauss) @ self._N_gauss   # (n_quads, 4)
            np.add.at(Q_i, self._quads_flat,
                      (self._n_exposed_sides * self._exp_2d * f_rad_elem).ravel())

            # Prescribed flux (uniform FireZone USERFLUX) — exposure-scaled
            if q_presc:
                np.add.at(Q_i, self._quads_flat,
                          (self._n_exposed_sides * q_presc
                           * self._exp_2d * self._M_consistent_rowsum).ravel())

            # Per-quad falloff flux (§3.5.4) — consistent row sums, face-dependent
            if self._q_per_quad is not None:
                q_dir = (np.asarray(self._q_quad_fn(t), dtype=float)
                         if self._q_quad_fn is not None else self._q_per_quad)
                np.add.at(Q_i, self._quads_flat,
                          (q_dir[:, None] * self._M_consistent_rowsum).ravel())

            # Steel surface re-radiation (§3.2.4) — Gauss-point integrated.
            # q_presc (uniform) sides scale by n_exposed_sides; q_per_quad (directional) does not.
            if self._eps_rerad > 0.0 and (q_presc or self._q_per_quad is not None):
                q_rerad_gp   = -self._eps_rerad * _SIGMA * T_gp_K ** 4  # (n_quads, 4)
                f_rerad_elem = (q_rerad_gp * self._det_J_gauss) @ self._N_gauss
                rerad_scale  = self._n_exposed_sides if q_presc else 1
                np.add.at(Q_i, self._quads_flat, (rerad_scale * f_rerad_elem).ravel())

        return K.tocsr(), M_i, Q_i

    def _init_rate(self, T0: np.ndarray, t0: float = 0.0) -> None:
        """
        Initialise CN state:  Ṫ0 = M0⁻¹ · (Q0 − K0 · T0)
        Prescribed DOFs (§3.5.2) have zero initial rate so the predictor
        T_iter = T_prev + dt·Ṫ_prev does not perturb them at the first step.
        """
        K0, M0, Q0 = self._assemble_step(T0, t0)
        if not sp.issparse(M0) and np.any(M0 == 0.0):
            raise ValueError(
                "Lumped mass matrix contains zero entries — material density (rho) "
                "is likely zero. Rigid/constraint materials (rho=0) cannot be "
                "solved thermally. Check the material assigned to this element."
            )
        self._K_prev     = K0
        self._M_prev     = M0
        T_dot = _mass_solve(M0, Q0 - K0 @ T0)
        # Zero prescribed-DOF rates so the CN predictor leaves them unchanged
        if self._prescribed_bcs:
            for bc in self._prescribed_bcs:
                for i in bc.node_indices:
                    T_dot[i] = 0.0
        self._T_dot_prev = T_dot

    def _nonlinear_converged(self, T_new: np.ndarray, T_iter: np.ndarray) -> bool:
        """Return True when the Picard iterate is converged in relative max norm."""
        delta = float(np.max(np.abs(T_new - T_iter)))
        scale = max(1.0, float(np.max(np.abs(T_new))))
        return delta <= self._nonlinear_tol * scale

    def _apply_dirichlet(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        t: float,
        T_prev: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        Apply prescribed nodal temperatures (§3.5.2) via Dirichlet elimination.

        The CN system solves for the **increment** ΔT = T_new − T_prev:
            A · ΔT = B
            T_new  = T_prev + ΔT

        To pin T_new[i] = T_presc, we need ΔT[i] = T_presc − T_prev[i].
        Dirichlet elimination sets:
            A[i, :] = 0,  A[i, i] = 1,  B[i] = T_presc − T_prev[i]

        After the solve: ΔT[i] = T_presc − T_prev[i] → T_new[i] = T_presc.

        The prescribed value is also clamped exactly after the solve (see `step`)
        to guard against floating-point rounding.

        Modifies ``b`` in-place; returns a new CSR matrix for ``A`` (CSR row
        modification requires a round-trip through LIL format).

        Args:
            A:      Assembled CN system matrix (n×n CSR), solving for ΔT.
            b:      Right-hand-side vector (n,), also for ΔT increments.
            t:      Current step-end time [s] — used to evaluate callable T.
            T_prev: Nodal temperatures from the previous step [°C].

        Returns:
            (A_mod, b_mod) with Dirichlet rows applied.
        """
        if not self._prescribed_bcs:
            return A, b

        A_lil = A.tolil()
        for bc in self._prescribed_bcs:
            T_val = bc.eval(t)
            for i in bc.node_indices:
                A_lil[i, :] = 0.0
                A_lil[i, i] = 1.0
                b[i] = T_val - T_prev[i]   # enforce ΔT[i] = T_presc − T_prev[i]
        return A_lil.tocsr(), b

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, T_prev: np.ndarray, dt: float, t: float) -> np.ndarray:
        """
        One Crank-Nicolson step ending at time t.

        Args:
            T_prev: Nodal temperatures from previous step [°C].
            dt:     Time step size [s].
            t:      Current time (end of step) [s].

        Returns:
            T_new: Updated nodal temperatures [°C].
        """
        if self._K_prev is None:
            self._init_rate(T_prev, t0=0.0)

        assert self._K_prev is not None
        assert self._M_prev is not None
        assert self._T_dot_prev is not None

        two_over_dt = 2.0 / dt
        T_iter = T_prev + dt * self._T_dot_prev
        dT: np.ndarray | None = None

        for _iter_i in range(self._nonlinear_max_iter):
            K_i, M_i, Q_i = self._assemble_step(T_iter, t)
            A = K_i + _mass_to_matrix(M_i, two_over_dt)
            # Current-iterate K_i, M_i on the history side (not K_prev/M_prev):
            # keeps CN 2nd-order in Δt when k(T)/c(T) vary.
            B = (
                Q_i - K_i @ T_prev
                + _mass_matvec(M_i, self._T_dot_prev)
            )

            # §3.5.2 Dirichlet elimination: pin prescribed DOFs before solve.
            # B is modified in-place; A_mod is a new CSR matrix.
            A_mod, B = self._apply_dirichlet(A.tocsr(), B, t, T_prev)
            dT = spla.spsolve(A_mod, B)
            T_new = T_prev + dT

            # Overwrite prescribed DOFs so they are exactly at the target value.
            # (Floating-point arithmetic may produce tiny residuals otherwise.)
            if self._prescribed_bcs:
                for bc in self._prescribed_bcs:
                    T_val = bc.eval(t)
                    for i in bc.node_indices:
                        T_new[i] = T_val

            if self._nonlinear_converged(T_new, T_iter):
                break
            T_iter = T_new

        assert dT is not None
        T_new     = T_prev + dT

        # Re-enforce prescribed values after final dT update (exact pinning).
        if self._prescribed_bcs:
            for bc in self._prescribed_bcs:
                T_val = bc.eval(t)
                for i in bc.node_indices:
                    T_new[i] = T_val

        T_dot_new = two_over_dt * dT - self._T_dot_prev

        # Zero out the rate at prescribed DOFs so the CN predictor does not
        # push them away from the prescribed temperature at the next step.
        if self._prescribed_bcs:
            for bc in self._prescribed_bcs:
                for i in bc.node_indices:
                    T_dot_new[i] = 0.0

        K_i, M_i, _Q_i = self._assemble_step(T_new, t)

        self._K_prev     = K_i
        self._M_prev     = M_i
        self._T_dot_prev = T_dot_new

        return T_new

    def run(
        self,
        t_end: float,
        dt: float,
        output_dt: float | None = None,
        callback: Callable[[float, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Full transient simulation from t = 0 to t = t_end.

        Returns:
            times:     (n_out,) output times [s].
            T_history: (n_out, n_nodes) nodal temperatures [°C].
        """
        n = self._mesh.n_nodes
        T = np.full(n, self._T0, dtype=float)
        for bc in self._prescribed_bcs:          # prescribed DOFs start at their t=0 value
            T[list(bc.node_indices)] = bc.eval(0.0)
        out_dt    = output_dt if output_dt is not None else dt
        out_every = max(1, round(out_dt / dt))

        self._K_prev = None
        self._init_rate(T, t0=0.0)

        out_times: list[float]      = [0.0]
        out_T:     list[np.ndarray] = [T.copy()]

        n_steps = max(1, round(t_end / dt))
        t = 0.0

        for step_i in range(n_steps):
            dt_step = min(dt, t_end - t)
            if dt_step <= 0.0:
                break
            t += dt_step
            T = self.step(T, dt_step, t)

            if callback is not None:
                callback(t, T)

            if (step_i + 1) % out_every == 0 or step_i == n_steps - 1:
                out_times.append(t)
                out_T.append(T.copy())

        return np.array(out_times), np.array(out_T)
