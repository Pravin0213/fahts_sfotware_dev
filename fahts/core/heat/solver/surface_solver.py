"""
Crank-Nicolson transient heat solver for a 3-D beam surface mesh.

Governing equation (2-D in face plane, axial × hoop):
    M · Ṫ + K · T = Q(T)

Element matrices for a surface element with face area A_e and wall thickness t:
    K_e = t × quad4_conductivity_matrix(coords_2d, k)   [W/K]
    M_e = t × quad4_capacity_matrix(coords_2d, ρ, cp)   [J/K]

Fire boundary condition applied to entire element face (2-D area integral):
    Q_conv = h_conv × T_fire × A_e/4 × [1,1,1,1]  (load, semi-implicit)
    K_conv = h_conv × A_e/4 × I₄                   (stiffness, diagonal)
    Q_rad  = ε·σ·(T_fire_K⁴ − T_K⁴) × A_e/4       (radiation, explicit)
    Q_presc = q_prescribed × A_e/4                  (prescribed flux)

Crank-Nicolson (θ=1/2) incremental form (SINTEF FAHTS Eq. 3.2.30):
    A · ΔTi = B
    A = Ki + (2/Δt) · Mi
    B = Qi − K_{i−1}·T_{i−1} + M_{i−1}·Ṫ_{i−1}
    Ti = T_{i−1} + ΔTi
    Ṫi = (2/Δt)·ΔTi − Ṫ_{i−1}

Note: Heat accumulation element for hollow BOX cavities is NOT implemented here
(no inner surface nodes exist in the surface mesh).  The inner air thermal mass
can be added as a future enhancement.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


def assemble_surface_M(mesh: BeamSurfaceMesh, rho: float, cp: float) -> np.ndarray:
    """
    Global lumped capacitance vector for a beam surface mesh.

    M_e = thickness × quad4_capacity_matrix(local_2d_coords, ρ, cp)

    Returns:
        (n_nodes,) diagonal of the lumped mass matrix [J/K].
    """
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

    The boundary condition is applied to the entire face area of each element
    (2-D area integral), not just a boundary edge.  Lumped distribution (A_e/4
    per node) is used for consistency with the lumped mass matrix.

    Convection (semi-implicit — adds to stiffness to improve stability):
        K[i,i] += h_conv × A_e / 4
        Q[i]   += h_conv × T_fire × A_e / 4

    Radiation (fully explicit, EN 1993-1-2 §3.1):
        Q[i] += ε_m × σ × (T_fire_K⁴ − T_i_K⁴) × A_e / 4

    Prescribed flux (RadiationBall / USERFLUX):
        Q[i] += q_prescribed × A_e / 4
    """
    T_fire_K = T_fire + 273.15

    for q, quad in enumerate(mesh.quads):
        coords_2d = mesh.element_coords_2d(q)
        A4 = _quad_area(coords_2d) * 0.25   # A_e / 4

        for nid in quad:
            nid = int(nid)
            # Convection (semi-implicit)
            K_lil[nid, nid] += h_conv * A4
            Q[nid] += h_conv * T_fire * A4

            # Radiation (explicit)
            T_K = T_prev[nid] + 273.15
            q_rad = epsilon_m * _SIGMA * (T_fire_K ** 4 - T_K ** 4)
            Q[nid] += q_rad * A4

            # Prescribed flux
            if q_prescribed:
                Q[nid] += q_prescribed * A4


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
    ) -> None:
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
        self._q_per_quad: np.ndarray | None = (
            np.asarray(q_per_quad, dtype=float) if q_per_quad is not None else None
        )

        # §3.2.4 steel surface re-radiation for prescribed-flux (RadiationBall) mode.
        # When epsilon_m > 0 (FireZone), q_rad = ε_m·σ·(T_fire⁴ − T_steel⁴) already
        # contains the -T_steel⁴ term.  When epsilon_m = 0 (prescribed flux), there is
        # no fire temperature and q_rad = 0, so re-radiation must be applied separately:
        #   Q_rerad[node] = −ε_steel · σ · T_node_K⁴ · A_e/4
        self._eps_rerad: float = (
            (epsilon_steel if epsilon_steel is not None else 0.7)
            if epsilon_m == 0.0 else 0.0
        )

        # CN state — initialised by _init_rate()
        self._K_prev:     sp.csr_matrix | None = None
        self._M_prev:     np.ndarray | None    = None
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

        # Flat quad-node indices for np.add.at BC assembly: (n_quads*4,)
        self._quads_flat: np.ndarray = mesh.quads.ravel().astype(np.intp)
        self._n = n

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assemble_step(
        self, T_prev: np.ndarray, t: float
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Vectorised assembly using pre-cached geometry.

        All Python-level quad loops are eliminated: K and M are built from
        pre-computed base matrices scaled by the current scalar k and rho*cp,
        then summed via COO → CSR.  The BC diagonal is added directly without
        a tolil/tocsr round-trip.
        """
        T_mean   = float(np.mean(T_prev))
        k        = self._mat.conductivity(T_mean)
        cp       = self._mat.specific_heat(T_mean)
        rho      = self._mat.rho
        T_fire   = self._fire_temp(t)
        T_fire_K = T_fire + 273.15
        q_presc  = self._q_fn(t)
        n        = self._n

        # ── Conductivity matrix (vectorised) ─────────────────────────────────
        # K_vals[q*16 + li*4 + lj] = k * t[q] * K_base[q, li, lj]
        K_vals = (k * self._t_arr[:, None, None] * self._K_base).ravel()
        K_i = sp.csr_matrix(
            (K_vals, (self._rows_K, self._cols_K)), shape=(n, n)
        )

        # ── Lumped capacity vector (vectorised) ───────────────────────────────
        M_vals = (rho * cp * self._t_arr[:, None] * self._M_base).ravel()
        M_i = np.zeros(n)
        np.add.at(M_i, self._quads_flat, M_vals)

        # ── Fire BC (vectorised, no tolil/tocsr) ──────────────────────────────
        # Convective stiffness diagonal
        K_bc_diag = np.zeros(n)
        np.add.at(K_bc_diag, self._quads_flat,
                  np.repeat(self._h_conv * self._areas_4, 4))

        Q_i = np.zeros(n)
        # Convection load
        np.add.at(Q_i, self._quads_flat,
                  np.repeat(self._h_conv * T_fire * self._areas_4, 4))

        # Radiation (explicit) — §3.2.4 with geometric view factor §3.3.4
        # With view factors: q = ε_steel·σ·(F·ε_fire·T_fire⁴ − T_node⁴)
        # Without (F=1):     q = ε_m·σ·(T_fire⁴ − T_node⁴)
        T_elem_K = (T_prev[self._mesh.quads] + 273.15)          # (n_quads, 4)
        if self._use_vf:
            q_rad = self._eps_steel * _SIGMA * (
                self._F_arr[:, None] * self._eps_fire * T_fire_K ** 4
                - T_elem_K ** 4
            )
        else:
            q_rad = self._eps * _SIGMA * (T_fire_K ** 4 - T_elem_K ** 4)
        np.add.at(Q_i, self._quads_flat,
                  (q_rad * self._areas_4[:, None]).ravel())

        # Prescribed flux (RadiationBall / USERFLUX) — uniform across all faces
        if q_presc:
            np.add.at(Q_i, self._quads_flat,
                      np.repeat(q_presc * self._areas_4, 4))

        # Per-quad falloff flux (§3.5.4) — directional, face-dependent
        if self._q_per_quad is not None:
            np.add.at(Q_i, self._quads_flat,
                      np.repeat(self._q_per_quad * self._areas_4, 4))

        # Steel surface re-radiation (§3.2.4) — only for prescribed-flux mode.
        # All element faces emit ε_steel·σ·T⁴ regardless of illumination direction.
        if self._eps_rerad > 0.0 and (q_presc or self._q_per_quad is not None):
            q_rerad = -self._eps_rerad * _SIGMA * T_elem_K ** 4  # (n_quads, 4)
            np.add.at(Q_i, self._quads_flat,
                      (q_rerad * self._areas_4[:, None]).ravel())

        K_total = K_i + sp.diags(K_bc_diag)
        return K_total.tocsr(), M_i, Q_i

    def _init_rate(self, T0: np.ndarray, t0: float = 0.0) -> None:
        """
        Initialise CN state:  Ṫ0 = M0⁻¹ · (Q0 − K0 · T0)
        """
        K0, M0, Q0 = self._assemble_step(T0, t0)
        if np.any(M0 == 0.0):
            raise ValueError(
                "Lumped mass matrix contains zero entries — material density (rho) "
                "is likely zero. Rigid/constraint materials (rho=0) cannot be "
                "solved thermally. Check the material assigned to this element."
            )
        self._K_prev     = K0
        self._M_prev     = M0
        self._T_dot_prev = (Q0 - K0 @ T0) / M0

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

        K_i, M_i, Q_i = self._assemble_step(T_prev, t)

        two_over_dt = 2.0 / dt
        A = K_i + sp.diags(two_over_dt * M_i)
        B = Q_i - self._K_prev @ T_prev + self._M_prev * self._T_dot_prev

        dT = spla.spsolve(A.tocsr(), B)
        T_new     = T_prev + dT
        T_dot_new = two_over_dt * dT - self._T_dot_prev

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
