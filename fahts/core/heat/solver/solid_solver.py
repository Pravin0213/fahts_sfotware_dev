"""
Crank-Nicolson transient heat solver for a 3-D Hex8 solid mesh (`SolidMesh`).

Governing equation (true 3-D conduction in the steel volume):
    ρ c(T) ∂T/∂t = ∇·(k(T) ∇T)                        in Ω
    −k ∂T/∂n = h (T − T_fire) + ε σ (T⁴ − T_fire⁴) − q  on Γ_outer (FACE_OUTER)
    −k ∂T/∂n = h_in(t) (T − T_fluid(t))                on Γ_inner if `inner_bc` else 0
    −k ∂T/∂n = 0                                       on Γ_end

Semi-discrete Galerkin system:  M(T) Ṫ + K(T) T = Q(T)
    K = Σ_hex k(T̄_e) ∫ Bᵀ B dV + Σ_faces h ∫ Nᵢ Nⱼ dA
    M = Σ_hex ρ c(T̄_e) ∫ Nᵢ Nⱼ dV (row-sum lumped by default) + M_extra
    Q = Σ_faces ∫ Nᵢ (h T_env + q_rad(T) + q_src) dA     (2×2 Gauss on faces)
k(T), c(T) are evaluated per hex at the element mean temperature T̄_e.

Time integration: Crank-Nicolson (θ = ½) incremental form with bounded Picard
iteration — identical to `SurfaceTransientSolver` so `GlobalThermalSolver`
can drive either solver through `_assemble_step`.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fahts.core.heat.bc.inner_robin_bc import InnerRobinBC
from fahts.core.heat.bc.insulation import InsulationLayer
from fahts.core.heat.bc.prescribed_node_bc import PrescribedNodeBC
from fahts.core.heat.solid_mesh.solid_mesh import FACE_INNER, FACE_OUTER, SolidMesh
from fahts.core.heat.solver.linear_solve import SPDSolver
from fahts.core.heat.solver.fem_3d import (
    hex8_capacity_base,
    hex8_conductivity_base,
    hex8_conductivity_twopoint_base,
    quad3d_face_gauss,
    quad3d_face_mass_base,
)
from fahts.core.model.material import SteelMaterial

logger = logging.getLogger(__name__)

_SIGMA = 5.67e-8   # Stefan-Boltzmann constant [W/(m²·K⁴)]
_K0 = 273.15

MassMatrix = np.ndarray | sp.csr_matrix

# Property tabulation grid for SteelMaterial (exact at the EC3 breakpoints)
_TAB_T_MIN = -50.0
_TAB_T_MAX = 1500.0
_TAB_DT = 0.1


# ── Temperature-dependent material properties (vectorised) ───────────────────

@lru_cache(maxsize=64)
def _steel_property_table(
    usfos_mode: bool, k_ref: float, c_ref: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate k(T), c(T) of a SteelMaterial on a fine grid (cached per parameters)."""
    mat = SteelMaterial(mid=0, E=0.0, nu=0.0, fy=0.0, rho=1.0, alpha_T=0.0,
                        usfos_mode=usfos_mode, k_ref=k_ref, c_ref=c_ref)
    n = int(round((_TAB_T_MAX - _TAB_T_MIN) / _TAB_DT)) + 1
    T = np.linspace(_TAB_T_MIN, _TAB_T_MAX, n)
    k = np.array([mat.conductivity(float(x)) for x in T])
    c = np.array([mat.specific_heat(float(x)) for x in T])
    return T, k, c


def _eval_scalar_fn(fn: Callable[[float], float], T: np.ndarray) -> np.ndarray:
    """Evaluate a scalar-only property function on unique (0.01 °C-rounded) temps."""
    Tr = np.round(np.asarray(T, dtype=float), 2)
    uniq, inv = np.unique(Tr, return_inverse=True)
    vals = np.array([float(fn(float(x))) for x in uniq])
    return vals[inv]


def material_properties(material: Any, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised k(T) [W/(m·K)] and c(T) [J/(kg·K)] for an array of temperatures.

    SteelMaterial (EC3 / USFOS) is served from a cached 0.1 °C lookup table
    with linear interpolation (exact for the piecewise-linear parts).  A material
    object with a vectorised ``properties(T_array) -> (k, c)`` method is called
    directly.  Any other material object exposing scalar ``conductivity(T)`` /
    ``specific_heat(T)`` is evaluated once per unique temperature (rounded to 0.01 °C).
    """
    T = np.asarray(T, dtype=float)
    if hasattr(material, "properties"):
        k, c = material.properties(T)
        return np.asarray(k, dtype=float), np.asarray(c, dtype=float)
    if isinstance(material, SteelMaterial):
        Tg, kg, cg = _steel_property_table(
            bool(material.usfos_mode), float(material.k_ref), float(material.c_ref)
        )
        return np.interp(T, Tg, kg), np.interp(T, Tg, cg)
    return (_eval_scalar_fn(material.conductivity, T),
            _eval_scalar_fn(material.specific_heat, T))


# ── Sparse assembly helper ────────────────────────────────────────────────────

class _CSRPattern:
    """
    Fixed CSR sparsity pattern for repeated COO → CSR assembly.

    Duplicate (row, col) entries are summed with a single `np.bincount`,
    avoiding scipy's COO sort on every assembly.
    """

    def __init__(self, rows: np.ndarray, cols: np.ndarray, n: int) -> None:
        lin = rows.astype(np.int64) * n + cols.astype(np.int64)
        uniq, self._inv = np.unique(lin, return_inverse=True)
        self._inv = self._inv.ravel()
        self._nnz = len(uniq)
        self._indices = (uniq % n).astype(np.int32)
        counts = np.bincount((uniq // n).astype(np.intp), minlength=n)
        self._indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self._n = n

    def data(self, vals: np.ndarray) -> np.ndarray:
        """CSR ``data`` array (pattern order) from the COO value stream."""
        return np.bincount(self._inv, weights=vals, minlength=self._nnz)

    def wrap(self, data: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix(
            (data, self._indices.copy(), self._indptr.copy()), shape=(self._n, self._n)
        )

    def build(self, vals: np.ndarray) -> sp.csr_matrix:
        return self.wrap(self.data(vals))

    @property
    def structure(self) -> tuple[np.ndarray, np.ndarray]:
        """(indptr, indices) of the fixed CSR pattern."""
        return self._indptr, self._indices


def _mass_to_matrix(M: MassMatrix, scale: float = 1.0) -> sp.csr_matrix:
    if sp.issparse(M):
        return (M * scale).tocsr()
    return sp.diags(scale * M, format="csr")


def _mass_matvec(M: MassMatrix, v: np.ndarray) -> np.ndarray:
    return M @ v if sp.issparse(M) else M * v


def _mass_solve(M: MassMatrix, rhs: np.ndarray) -> np.ndarray:
    if sp.issparse(M):
        return spla.spsolve(M.tocsc(), rhs)
    return rhs / M


def _per_face(arr: np.ndarray | None, n: int, name: str) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},) (one per FACE_OUTER face), "
                         f"got {a.shape}")
    return a


# ── Solver ────────────────────────────────────────────────────────────────────

class SolidTransientSolver:
    """
    Crank-Nicolson (θ = ½) 3-D Hex8 FEM heat solver for one solid member.

    Public surface mirrors `SurfaceTransientSolver` (`_assemble_step`,
    `_init_rate`, `step`, `run`).  All per-face arrays are indexed in the
    order of `mesh.outer_face_indices` (one entry per FACE_OUTER face).

    Args:
        mesh:            SolidMesh (Hex8, VTK ordering, outward CCW faces).
        material:        Provides conductivity(T), specific_heat(T), rho.
        fire_temp:       Callable(t [s]) → T_fire [°C].
        epsilon_m:       Resultant emissivity; 0 → prescribed-flux mode with
                         steel re-radiation (ε_steel, default 0.7).
        h_conv:          Convective coefficient [W/(m²·K)] on FACE_OUTER.
        T0:              Initial uniform temperature [°C].
        q_prescribed_fn: Optional Callable(t) → uniform flux [W/m²] on FACE_OUTER.
        epsilon_steel:   Steel surface emissivity (view-factor / re-radiation mode).
        epsilon_fire:    Fire emissivity (view-factor mode, default 1).
        view_factors:    Optional (n_outer,) view factors → q_rad =
                         ε_s σ (F ε_f T_fire⁴ − T⁴).
        q_per_face:      Optional (n_outer,) static directional flux [W/m²].
        face_exposure:   Optional (n_outer,) fire-zone exposure ∈ [0, 1]; scales
                         convection, radiation and uniform prescribed flux.
        M_extra:         Optional (n_nodes,) extra lumped capacity [J/K].
        mass_matrix:     "lumped" (default) or "consistent".
        nonlinear_max_iter, nonlinear_tol: Picard iteration controls.
        insulation:      Optional InsulationLayer: h_ins replaces h_conv on
                         FACE_OUTER, radiation suppressed.
        prescribed_node_bcs: Optional Dirichlet node constraints (mesh node indices).
        inner_bc:        Optional InnerRobinBC on FACE_INNER (None = adiabatic).
        linear_solver:   "cg" (default, Jacobi-PCG, warm-started) or "direct" (SuperLU).
        conduction:      "consistent" (Galerkin trilinear K) or "monotone" (edge-based
                         two-point stencil, M-matrix — no over/undershoot on thin walls;
                         see fem_3d.hex8_conductivity_twopoint_base).
    """

    def __init__(
        self,
        mesh: SolidMesh,
        material: SteelMaterial,
        fire_temp: Callable[[float], float],
        epsilon_m: float,
        h_conv: float,
        T0: float = 20.0,
        q_prescribed_fn: Callable[[float], float] | None = None,
        epsilon_steel: float | None = None,
        epsilon_fire: float | None = None,
        view_factors: np.ndarray | None = None,
        q_per_face: np.ndarray | None = None,
        face_exposure: np.ndarray | None = None,
        M_extra: np.ndarray | None = None,
        mass_matrix: str = "lumped",
        nonlinear_max_iter: int = 6,
        nonlinear_tol: float = 1e-6,
        insulation: InsulationLayer | None = None,
        prescribed_node_bcs: list[PrescribedNodeBC] | None = None,
        inner_bc: InnerRobinBC | None = None,
        linear_solver: str = "cg",
        conduction: str = "consistent",
    ) -> None:
        if nonlinear_max_iter < 1:
            raise ValueError("nonlinear_max_iter must be >= 1")
        if nonlinear_tol < 0.0:
            raise ValueError("nonlinear_tol must be >= 0")
        if mass_matrix not in {"lumped", "consistent"}:
            raise ValueError("mass_matrix must be 'lumped' or 'consistent'")
        if conduction not in {"consistent", "monotone"}:
            raise ValueError("conduction must be 'consistent' or 'monotone'")
        self._conduction = conduction

        self._mesh = mesh
        self._mat = material
        self._fire_temp = fire_temp
        self._eps = float(epsilon_m)
        self._h_conv = float(h_conv)
        self._T0 = float(T0)
        self._q_fn: Callable[[float], float] = (
            q_prescribed_fn if q_prescribed_fn is not None else lambda _t: 0.0
        )

        self._outer = mesh.outer_face_indices
        self._inner = mesh.inner_face_indices
        n_o = len(self._outer)
        n = mesh.n_nodes

        self._F_arr = _per_face(view_factors, n_o, "view_factors")
        self._use_vf = self._F_arr is not None
        if self._use_vf:
            self._eps_steel = epsilon_steel if epsilon_steel is not None else epsilon_m
            self._eps_fire = epsilon_fire if epsilon_fire is not None else 1.0
        else:
            self._eps_steel = epsilon_m
            self._eps_fire = 1.0
        # q_per_face may be a static array or a callable t → (n_outer,) (time-varying
        # point/line-source power — see fahts/core/heat/bc/face_flux.py)
        self._q_face_fn: Callable[[float], np.ndarray] | None = (
            q_per_face if callable(q_per_face) else None
        )
        self._q_per_face = _per_face(
            q_per_face(0.0) if self._q_face_fn is not None else q_per_face, n_o, "q_per_face"
        )
        exp = _per_face(face_exposure, n_o, "face_exposure")
        self._exp = exp if exp is not None else np.ones(n_o)

        self._M_extra = np.asarray(M_extra, dtype=float) if M_extra is not None else None
        if self._M_extra is not None and self._M_extra.shape != (n,):
            raise ValueError(f"M_extra must have shape ({n},), got {self._M_extra.shape}")
        self._mass_matrix = mass_matrix
        self._nonlinear_max_iter = int(nonlinear_max_iter)
        self._nonlinear_tol = float(nonlinear_tol)

        # Steel re-radiation for prescribed-flux mode (no fire temperature)
        self._eps_rerad: float = (
            (epsilon_steel if epsilon_steel is not None else 0.7)
            if epsilon_m == 0.0 else 0.0
        )
        self._insulation = insulation
        self._inner_bc = inner_bc

        self._prescribed_bcs: list[PrescribedNodeBC] = (
            list(prescribed_node_bcs) if prescribed_node_bcs else []
        )
        idx = [i for bc in self._prescribed_bcs for i in bc.node_indices]
        for i in idx:
            if i < 0 or i >= n:
                raise ValueError(
                    f"PrescribedNodeBC node index {i} is out of range for "
                    f"mesh with {n} nodes."
                )
        self._free_mask = np.ones(n)
        if idx:
            self._free_mask[np.asarray(idx, dtype=np.intp)] = 0.0
        self._pinned = np.unique(np.asarray(idx, dtype=np.intp))
        self._free = np.flatnonzero(self._free_mask)
        self._linear = SPDSolver(linear_solver)

        # CN state
        self._K_prev: sp.csr_matrix | None = None
        self._M_prev: MassMatrix | None = None
        self._T_dot_prev: np.ndarray | None = None

        self._precompute_geometry()

    # ── Geometry pre-computation ──────────────────────────────────────────────

    def _precompute_geometry(self) -> None:
        """Cache all time-invariant element / face integrals and the CSR pattern."""
        mesh = self._mesh
        n = mesh.n_nodes
        hexes = np.asarray(mesh.hexes, dtype=np.intp)
        X = np.asarray(mesh.nodes, dtype=float)[hexes]                 # (n_h, 8, 3)
        self._n = n
        self._hexes = hexes
        self._hexes_flat = hexes.ravel()

        self._K_base = (hex8_conductivity_twopoint_base(X) if self._conduction == "monotone"
                        else hex8_conductivity_base(X))                # (n_h, 8, 8)
        if self._mass_matrix == "consistent":
            self._C_base = hex8_capacity_base(X, lumped=False)         # (n_h, 8, 8)
        else:
            self._C_base = hex8_capacity_base(X, lumped=True)          # (n_h, 8)

        faces = np.asarray(mesh.faces, dtype=np.intp)
        self._fo = faces[self._outer]                                   # (n_o, 4)
        self._fi = faces[self._inner]                                   # (n_i, 4)
        Po = np.asarray(mesh.nodes, dtype=float)[self._fo]
        Pi = np.asarray(mesh.nodes, dtype=float)[self._fi]
        self._Fo_mass = quad3d_face_mass_base(Po)                       # (n_o, 4, 4)
        self._Fo_row = self._Fo_mass.sum(axis=2)                        # (n_o, 4) ∫Nᵢ dA
        self._N_gp, self._Fo_detJ = quad3d_face_gauss(Po)               # (4,4), (n_o,4)
        self._Fi_mass = quad3d_face_mass_base(Pi)
        self._Fi_row = self._Fi_mass.sum(axis=2)
        if self._conduction == "monotone":
            # Lumped (nodal-quadrature) boundary terms, as in a finite-volume code: the
            # consistent face mass ∫NᵢNⱼ dA puts POSITIVE couplings into K for Robin and
            # radiation terms, which breaks the M-matrix property of the two-point
            # conduction.  Quadrature points = the face nodes, weights = ∫Nᵢ dA.
            eye = np.eye(4)
            self._Fo_mass = self._Fo_row[:, :, None] * eye
            self._Fi_mass = self._Fi_row[:, :, None] * eye
            self._N_gp = eye
            self._Fo_detJ = self._Fo_row.copy()

        # One combined sparsity pattern: hexes (64) + outer faces (16) + inner (16)
        def _pairs(conn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            m = conn.shape[1]
            return np.repeat(conn, m, axis=1).ravel(), np.tile(conn, (1, m)).ravel()

        rh, ch = _pairs(hexes)
        ro, co = _pairs(self._fo)
        ri, ci = _pairs(self._fi)
        self._K_pattern = _CSRPattern(
            np.concatenate([rh, ro, ri]), np.concatenate([ch, co, ci]), n
        )
        if self._mass_matrix == "consistent":
            self._M_pattern: _CSRPattern | None = _CSRPattern(rh, ch, n)
        else:
            self._M_pattern = None
        self._fo_flat = self._fo.ravel()
        self._fo_nodes = np.unique(self._fo_flat)
        self._fi_flat = self._fi.ravel()

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _assemble_step(
        self, T_prev: np.ndarray, t: float
    ) -> tuple[sp.csr_matrix, MassMatrix, np.ndarray]:
        """
        Assemble K (csr), M (lumped vector or csr) and Q at temperature iterate T.

        k(T), c(T) are evaluated per hex at the hex mean temperature.
        """
        K_data, M_i, Q = self.assemble_raw(T_prev, t)
        return self._K_pattern.wrap(K_data), M_i, Q

    def k_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """(indptr, indices) of K's fixed CSR pattern — for global scatter maps."""
        return self._K_pattern.structure

    def assemble_raw(
        self, T_prev: np.ndarray, t: float
    ) -> tuple[np.ndarray, MassMatrix, np.ndarray]:
        """As ``_assemble_step`` but K is returned as its CSR ``data`` array in the
        fixed ``k_structure()`` order (no scipy object construction)."""
        T = np.asarray(T_prev, dtype=float)
        n = self._n
        T_hex = T[self._hexes].mean(axis=1)                              # (n_h,)
        k_h, c_h = material_properties(self._mat, T_hex)
        rho = float(self._mat.rho)
        T_fire = float(self._fire_temp(t))
        T_fire_K = T_fire + _K0
        q_presc = float(self._q_fn(t))

        # Boundary conductance on FACE_OUTER
        if self._insulation is not None:
            T_surf = float(T[self._fo_nodes].mean()) if len(self._fo_nodes) else T_fire
            h_bc = self._insulation.conductance(0.5 * (T_fire + T_surf))
        else:
            h_bc = self._h_conv
        if self._inner_bc is not None:
            h_in = self._inner_bc.h_at(t)
            T_in = self._inner_bc.T_fluid_at(t)
        else:
            h_in, T_in = 0.0, 0.0


        # ── M: capacity ───────────────────────────────────────────────────────
        rc = rho * c_h
        if self._M_pattern is not None:
            M_i: MassMatrix = self._M_pattern.build((rc[:, None, None] * self._C_base).ravel())
            if self._M_extra is not None:
                M_i = (M_i + sp.diags(self._M_extra)).tocsr()
        else:
            M_i = np.bincount(self._hexes_flat, weights=(rc[:, None] * self._C_base).ravel(),
                              minlength=n)
            if self._M_extra is not None:
                M_i = M_i + self._M_extra

        # ── Q: face loads ─────────────────────────────────────────────────────
        rad_K: np.ndarray | None = None
        exp2 = self._exp[:, None]
        q_face = h_bc * T_fire * exp2 * self._Fo_row                     # (n_o, 4)

        if self._insulation is None and len(self._fo):
            if q_presc:
                q_face = q_face + q_presc * exp2 * self._Fo_row
            if self._q_per_face is not None:
                q_dir = (np.asarray(self._q_face_fn(t), dtype=float)
                         if self._q_face_fn is not None else self._q_per_face)
                q_face = q_face + q_dir[:, None] * self._Fo_row

            need_rad = self._use_vf or self._eps != 0.0
            # The radiation term already contains the −εσT⁴ steel emission, so
            # separate re-radiation is only added when no radiation term is active.
            need_rerad = not need_rad and self._eps_rerad > 0.0 and (
                bool(q_presc) or self._q_per_face is not None
            )
            if need_rad or need_rerad:
                T_gpK = (T[self._fo] + _K0) @ self._N_gp.T              # (n_o, 4gp)
                T_gp4 = T_gpK ** 4
                q_gp = np.zeros_like(T_gp4)
                # effective steel emissivity multiplying −σT⁴ at each Gauss point
                e_gp = np.zeros_like(T_gp4)
                if self._use_vf:
                    q_gp += exp2 * self._eps_steel * _SIGMA * (
                        self._F_arr[:, None] * self._eps_fire * T_fire_K ** 4 - T_gp4
                    )
                    e_gp += exp2 * self._eps_steel
                elif self._eps != 0.0:
                    q_gp += exp2 * self._eps * _SIGMA * (T_fire_K ** 4 - T_gp4)
                    e_gp += exp2 * self._eps
                if need_rerad:
                    # re-radiation to the ambient (initial) temperature, not to 0 K
                    q_gp -= self._eps_rerad * _SIGMA * (T_gp4 - (self._T0 + _K0) ** 4)
                    e_gp += self._eps_rerad
                # Newton linearisation of the −eσT⁴ emission about the iterate T*:
                #   q(T) ≈ q(T*) − h_r (T − T*),  h_r = 4 e σ T*³   (per Gauss point)
                # → K += ∫ h_r NᵢNⱼ dA,  Q += ∫ Nᵢ (q(T*) + h_r T*) dA.
                # Identical equation at convergence; converges far faster than lagging
                # T⁴ (the dominant nonlinearity in fire) and is unconditionally stable.
                h_r = 4.0 * e_gp * _SIGMA * T_gpK ** 3 * self._Fo_detJ    # (n_o, gp)
                q_gp = q_gp + (4.0 * e_gp * _SIGMA * T_gpK ** 3) * (T_gpK - _K0)
                q_face = q_face + (q_gp * self._Fo_detJ) @ self._N_gp
                rad_K = np.einsum("fg,gi,gj->fij", h_r, self._N_gp, self._N_gp)

        # ── K: conduction + Robin face terms (+ radiation tangent), one CSR pass ─
        face_K = h_bc * self._exp[:, None, None] * self._Fo_mass
        if rad_K is not None:
            face_K = face_K + rad_K
        K_vals = np.concatenate([
            (k_h[:, None, None] * self._K_base).ravel(),
            face_K.ravel(),
            (h_in * self._Fi_mass).ravel(),
        ])
        K = self._K_pattern.data(K_vals)

        Q = np.bincount(self._fo_flat, weights=q_face.ravel(), minlength=n)
        if h_in != 0.0 and len(self._fi):
            Q += np.bincount(self._fi_flat, weights=(h_in * T_in * self._Fi_row).ravel(),
                             minlength=n)
        return K, M_i, Q

    # ── CN machinery ──────────────────────────────────────────────────────────

    def _prescribed_values(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        idx: list[int] = []
        vals: list[float] = []
        for bc in self._prescribed_bcs:
            v = bc.eval(t)
            idx.extend(bc.node_indices)
            vals.extend([v] * len(bc.node_indices))
        return np.asarray(idx, dtype=np.intp), np.asarray(vals, dtype=float)

    def _init_rate(self, T0: np.ndarray, t0: float = 0.0) -> None:
        """Initialise CN state: Ṫ0 = M0⁻¹ (Q0 − K0 T0); prescribed DOFs get Ṫ = 0."""
        K0, M0, Q0 = self._assemble_step(T0, t0)
        if not sp.issparse(M0) and np.any(M0 == 0.0):
            raise ValueError(
                "Lumped mass matrix contains zero entries — material density (rho) "
                "is likely zero or nodes are not attached to any hex."
            )
        self._K_prev = K0
        self._M_prev = M0
        rhs = Q0 - K0 @ T0
        if self._prescribed_bcs and sp.issparse(M0):
            # Prescribed DOFs have Ṫ = 0: solve only the free block M_ff·Ṫ_f = rhs_f so
            # consistent-mass coupling to pinned rows cannot corrupt the free rates.
            free = np.flatnonzero(self._free_mask)
            T_dot = np.zeros_like(rhs)
            T_dot[free] = spla.spsolve(M0.tocsr()[free][:, free].tocsc(), rhs[free])
        else:
            T_dot = _mass_solve(M0, rhs)
            if self._prescribed_bcs:
                T_dot = T_dot * self._free_mask
        self._T_dot_prev = T_dot

    def _nonlinear_converged(self, T_new: np.ndarray, T_iter: np.ndarray) -> bool:
        delta = float(np.max(np.abs(T_new - T_iter)))
        scale = max(1.0, float(np.max(np.abs(T_new))))
        return delta <= self._nonlinear_tol * scale

    def _pin(self, T: np.ndarray, t: float) -> None:
        if self._prescribed_bcs:
            idx, vals = self._prescribed_values(t)
            T[idx] = vals

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mesh(self) -> SolidMesh:
        return self._mesh

    def step(self, T_prev: np.ndarray, dt: float, t: float) -> np.ndarray:
        """One Crank-Nicolson step ending at time *t*; returns new nodal temperatures."""
        if self._T_dot_prev is None:
            self._init_rate(T_prev, t0=0.0)
        assert self._T_dot_prev is not None

        two_over_dt = 2.0 / dt
        # Prescribed DOFs: known increment, removed by symmetric elimination (A stays SPD)
        dT_p = np.zeros(len(self._pinned))
        if len(self._pinned):
            target = T_prev.copy()
            self._pin(target, t)
            dT_p = target[self._pinned] - T_prev[self._pinned]

        dT = dt * self._T_dot_prev                 # predictor, also the CG warm start
        if len(self._pinned):
            dT[self._pinned] = dT_p
        T_iter = T_prev + dT
        for _ in range(self._nonlinear_max_iter):
            K_i, M_i, Q_i = self._assemble_step(T_iter, t)
            A = (K_i + _mass_to_matrix(M_i, two_over_dt)).tocsr()
            # CN with the CURRENT iterate's matrices on both sides:
            #   M_i(2ΔT/Δt − Ṫ_prev) + K_i(T_prev + ΔT) = Q_i
            # Using K_{i−1}, M_{i−1} on the history side (SINTEF Eq. 3.2.30) leaves an
            # O(Δt) error when k(T)/c(T) vary — validation case 7 showed order ≈ 1.
            B = Q_i - (K_i @ T_prev) + _mass_matvec(M_i, self._T_dot_prev)
            dT = self._linear.solve_constrained(A, B, self._free, self._pinned, dT_p, x0=dT)
            T_new = T_prev + dT
            if self._nonlinear_converged(T_new, T_iter):
                break
            T_iter = T_new

        T_new = T_prev + dT
        self._pin(T_new, t)
        T_dot_new = two_over_dt * dT - self._T_dot_prev
        if len(self._pinned):
            T_dot_new[self._pinned] = 0.0
        # No end-of-step reassembly: the history terms use the current iterate's K, M.
        self._K_prev, self._M_prev = K_i, M_i
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
        Full transient from t = 0 to t_end.

        Returns:
            times:     (n_out,) output times [s].
            T_history: (n_out, n_nodes) nodal temperatures [°C].
        """
        T = np.full(self._n, self._T0, dtype=float)
        self._pin(T, 0.0)
        out_every = max(1, round((output_dt if output_dt is not None else dt) / dt))

        self._T_dot_prev = None
        self._init_rate(T, t0=0.0)

        out_times: list[float] = [0.0]
        out_T: list[np.ndarray] = [T.copy()]
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
        logger.debug("SolidTransientSolver.run: %d steps, %d nodes", n_steps, self._n)
        return np.array(out_times), np.array(out_T)
