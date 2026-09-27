"""
Analysis orchestration — FAHTS heat transfer solver.

Outer loop: time steps (global CN advance for all elements simultaneously).
Inner loop: per-element assembly, parallelised with threads.

The global thermal system M·Ṫ + K·T = Q is assembled from all exposed
structural elements into a block-diagonal sparse system (one spsolve per
Picard iterate per time step).  Since structural elements do not share
thermal DOFs, the system is block-diagonal and mathematically equivalent
to N independent element solves — but provides infrastructure for future
element coupling at structural nodes.

Supports all structural element and section types using FAHTS axial × hoop
surface-shell approach (SINTEF FAHTS §3.2.2):
  - BOX      → BoxSurfaceMesher      + SurfaceTransientSolver
  - IHPROFIL → IProfileSurfaceMesher + SurfaceTransientSolver
  - PIPE     → PipeSurfaceMesher     + SurfaceTransientSolver
  - Shell    → PlateSurfaceMesher    + SurfaceTransientSolver  (QUADSHEL only)
               ShellMesher           + Shell1DSolver           (TRISHELL fallback)

3-D solid path (``config.solver_dim == "3d"``, default): every BOX / IHPROFIL / PIPE
member and QUADSHEL plate is meshed with Hex8 bricks (``n_layers_3d`` through the wall)
and solved by ``SolidTransientSolver``; boundary data are evaluated per FACE_OUTER face
with true outward normals (see ``solid_integration.py``).  Members are coupled by safe
node merging plus joint conductance links (``k_link``).  TRISHELL keeps the 1-D fallback.

Supports heat sources:
  - FireZone          — rectangular zone with fire curve (ISO 834, HC, user-defined)
  - RadiationBall     — spherical source, exact point-to-sphere view factor:
                        q = flux (engulfed, d<=radius) or flux*(radius/d)^2*cos(θ) (exterior)
  - ConcentratedSource — omnidirectional point source §3.5.4: q=E·cos(θ)/(4π·r²)
  - LineSource        — finite line source §3.5.5: n discrete sub-sources along segment
"""
from __future__ import annotations

import dataclasses
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fahts.core.heat.bc.face_flux import TimeVaryingFaceFlux
from fahts.core.heat.bc.view_factor import (
    element_quad_exposure_flags,
    exposed_element_ids,
    geometric_view_factor_double_area,
)
from fahts.core.heat.section_mesh.beam_surface_mesh import BeamSurfaceMesh
from fahts.core.heat.section_mesh.box_surface_mesher import BoxSurfaceMesher
from fahts.core.heat.section_mesh.iprofil_surface_mesher import IProfileSurfaceMesher
from fahts.core.heat.section_mesh.pipe_surface_mesher import PipeSurfaceMesher
from fahts.core.heat.section_mesh.plate_surface_mesher import PlateSurfaceMesher
from fahts.core.heat.section_mesh.shell_mesh import ShellMesher
from fahts.core.heat.solid_mesh import PlateSolidMesher, SolidMesh
from fahts.core.heat.solver import solid_integration as _si
from fahts.core.heat.solver.batch_assembly import SolidBatch, is_batchable
from fahts.core.heat.solver.linear_solve import SPDSolver
from fahts.core.heat.solver.solid_solver import SolidTransientSolver
from fahts.core.heat.solver.surface_solver import SurfaceTransientSolver
from fahts.core.heat.solver.shell_1d_solver import Shell1DSolver
from fahts.core.heat.sources.concentrated_source import ConcentratedSource
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.heat.sources.line_source import LineSource
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.model.fem_model import FEMModel
from fahts.core.model.section import BoxSection, ISection, PipeSection, PlateSection
from fahts.core.results.analysis_config import AnalysisConfig
from fahts.core.results.temperature_field import TemperatureField

log = logging.getLogger(__name__)

# Minimum flux [W/m²] below which an element is not considered exposed by a
# RadiationBall.  The exact point-to-sphere law decays smoothly to zero with
# distance but never reaches it exactly, so this cutoff keeps the solver from
# processing elements receiving negligible radiation far from the ball.
_MIN_BALL_FLUX: float = 1.0

# Thermal density [kg/m³] substituted for rigid "dummy" materials that declare
# mechanical density rho=0 via MISOIEP (rigid links / connection members).
# USFOS still heats these elements thermally using the thermpar steel density,
# so we do the same for benchmark agreement.  See docs/3D_FEM_heat_transfer_theory.txt §16.
_THERMAL_DENSITY_FALLBACK: float = 7850.0

# Steel surface re-radiation emissivity used in USFOS benchmark mode.
# Matches the ``emiss = 0.85`` value declared in usfos_verification_results/fahts.fem;
# the production default re-radiation emissivity is 0.7 (see surface_solver.py).
_USFOS_BENCHMARK_EMISSIVITY: float = 0.85


class _NullPool:
    """Stand-in for a ThreadPoolExecutor when running single-threaded."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None


class AnalysisCancelledError(Exception):
    """Raised by run_analysis when the cancel_check callback returns True."""


class GlobalThermalSolver:
    """
    Global thermal system for all exposed structural elements with co-located node merging.

    Co-located mesh nodes from different elements (within the merge tolerance in global
    space) are merged into shared global DOFs so heat flows between connected elements.
    The assembled K and M are sparse matrices in global DOF space.  An optional constant
    joint-link conductance matrix ``k_link`` (3-D solid path — see
    ``solid_integration.build_joint_links``) is added to K at every assembly.

    Crank-Nicolson (θ=1/2) incremental form:
        A = K_i + (2/Δt) · M_i
        B = Q_i − K_i · T_prev + M_i · Ṫ_prev   (current Picard iterate)
        ΔT = A⁻¹ B        (direct spsolve, or Jacobi-PCG when linear_solver="cg")
        T_new    = T_prev + ΔT
        Ṫ_new    = (2/Δt) · ΔT − Ṫ_prev

    Element matrices are scattered as COO triplets (never densified), so members with
    thousands of solid nodes assemble in O(nnz).

    Linear solves: Jacobi-PCG by default with relative residual 1e-8 (≈1e-6 K solution
    error on the 64k-DOF model_file.fem system — far below the 1e-6·T Picard tolerance;
    1e-10 costs twice the iterations for no visible change).  ``linear_solver="direct"``
    uses SuperLU.

    Prescribed nodal temperatures (§3.5.2 ``PrescribedNodeBC`` held by the member
    solvers in local node numbering) are mapped to global DOFs and enforced by
    symmetric elimination: only the free block A_ff·ΔT_f = B_f − A_fp·ΔT_p is solved,
    so A stays SPD (valid for CG) and pinned DOFs carry Ṫ = 0.
    """

    def __init__(
        self,
        eids: list[int],
        solvers: dict,
        gdof_map: dict[int, np.ndarray],
        n_global_dofs: int,
        n_workers: int = 1,
        nonlinear_max_iter: int = 6,
        nonlinear_tol: float = 1e-6,
        linear_solver: str = "direct",
        k_link: sp.csr_matrix | None = None,
        cg_rtol: float = 1e-8,
        cg_maxiter: int | None = None,
        batch: bool = True,
        rad_exchange=None,
    ) -> None:
        if linear_solver not in {"direct", "cg"}:
            raise ValueError(f"linear_solver must be 'direct' or 'cg', got {linear_solver!r}")
        self._eids       = eids
        self._solvers    = solvers
        self._gdof_map   = gdof_map
        self._n_total    = n_global_dofs
        self._n_workers  = max(1, n_workers)
        self._max_iter   = nonlinear_max_iter
        self._tol        = nonlinear_tol
        self._linear     = linear_solver
        self._cg_rtol    = float(cg_rtol)
        self._cg_maxiter = cg_maxiter
        if k_link is not None:
            if k_link.shape != (n_global_dofs, n_global_dofs):
                raise ValueError(
                    f"k_link shape {k_link.shape} != ({n_global_dofs}, {n_global_dofs})"
                )
            k_link = sp.csr_matrix(k_link)
        self._k_link = k_link
        self._k_link_coo = k_link.tocoo() if k_link is not None else None
        # concatenated local→global DOF map (member order) for bincount scatters
        self._gd_concat = np.concatenate([np.asarray(gdof_map[e], dtype=np.intp) for e in eids])
        self._gd_offsets = np.cumsum([0] + [len(gdof_map[e]) for e in eids])[:-1]
        self._spd = SPDSolver(linear_solver, rtol=cg_rtol, maxiter=cg_maxiter)
        # optional surface-to-surface radiation exchange (radiation/exchange.py)
        self._rad_exchange = rad_exchange
        # Batchable solid members are assembled together (SolidBatch); the rest one by one
        batch_eids = [e for e in eids if is_batchable(solvers[e])] if batch else []
        self._other_eids = [e for e in eids if e not in set(batch_eids)]
        self._batch = (SolidBatch(batch_eids, solvers, gdof_map, n_global_dofs)
                       if batch_eids else None)
        self._gd_other = (np.concatenate([np.asarray(gdof_map[e], dtype=np.intp)
                                          for e in self._other_eids])
                          if self._other_eids else np.zeros(0, dtype=np.intp))
        # raw fast path when every remaining member exposes assemble_raw/k_structure
        self._raw = all(hasattr(solvers[e], "assemble_raw") and hasattr(solvers[e], "k_structure")
                        for e in self._other_eids)

        # Prescribed DOFs: [(global_dofs, PrescribedNodeBC)] — evaluated per step
        self._presc: list[tuple[np.ndarray, object]] = []
        for eid in eids:
            for bc in getattr(solvers[eid], "_prescribed_bcs", None) or []:
                local = np.asarray(bc.node_indices, dtype=np.intp)
                self._presc.append((np.asarray(gdof_map[eid])[local], bc))
        pinned = (np.unique(np.concatenate([d for d, _ in self._presc]))
                  if self._presc else np.zeros(0, dtype=np.intp))
        self._pinned = pinned
        self._free = np.setdiff1d(np.arange(n_global_dofs), pinned)

        self._K_prev:     sp.csr_matrix | None = None
        self._M_prev:     sp.csr_matrix | None = None
        self._T_dot_prev: np.ndarray | None    = None
        # Cached global sparsity pattern (rows/cols of the concatenated COO stream)
        self._pat_key: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pat:     dict[str, object] = {}

    @staticmethod
    def _to_sparse_K(K_e) -> sp.csr_matrix:
        return K_e.tocsr() if sp.issparse(K_e) else sp.csr_matrix(np.asarray(K_e))

    @staticmethod
    def _to_sparse_M(M_e) -> sp.csr_matrix:
        if sp.issparse(M_e):
            return M_e.tocsr()
        arr = np.asarray(M_e)
        return sp.diags(arr, format="csr") if arr.ndim == 1 else sp.csr_matrix(arr)

    @staticmethod
    def _coo_triplets(
        A_e, gdofs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Element matrix (sparse, dense 2-D or lumped 1-D) → global COO triplets."""
        if sp.issparse(A_e) and A_e.format == "csr":
            # Direct CSR expansion (avoids tocoo()'s canonical-format check)
            rows = np.repeat(np.arange(A_e.shape[0]), np.diff(A_e.indptr))
            return gdofs[rows], gdofs[A_e.indices], np.asarray(A_e.data, dtype=float)
        if sp.issparse(A_e):
            c = A_e.tocoo()
            return gdofs[c.row], gdofs[c.col], np.asarray(c.data, dtype=float)
        arr = np.asarray(A_e, dtype=float)
        if arr.ndim == 1:
            return gdofs, gdofs, arr
        c = sp.coo_matrix(arr)
        return gdofs[c.row], gdofs[c.col], np.asarray(c.data, dtype=float)

    def _build(
        self, name: str, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
    ) -> sp.csr_matrix:
        """COO → CSR, reusing a cached pattern when the triplet layout is unchanged."""
        from fahts.core.heat.solver.solid_solver import _CSRPattern

        key = self._pat_key.get(name)
        if (key is None or len(key[0]) != len(rows)
                or not np.array_equal(key[0], rows) or not np.array_equal(key[1], cols)):
            self._pat_key[name] = (rows, cols)
            self._pat[name] = _CSRPattern(rows, cols, self._n_total)
        return self._pat[name].build(vals)  # type: ignore[attr-defined]

    def _assemble_global(
        self, T_global: np.ndarray, t: float, pool
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """Global K, M and Q from all members via the global DOF map.

        Co-located nodes that share a global DOF accumulate contributions from all
        attached elements, producing off-diagonal coupling between elements.

        Batchable solid members (lumped mass, no insulation) are assembled together by
        ``SolidBatch`` in a few vectorised operations; all other members are assembled
        one by one.  The global CSR pattern (batch block + other members + joint links)
        is built once; each assembly is then a single ``bincount`` scatter.
        """
        n = self._n_total
        raw = self._raw

        def _one(eid: int):
            gdofs = self._gdof_map[eid]
            sv = self._solvers[eid]
            return (sv.assemble_raw(T_global[gdofs], t) if raw
                    else sv._assemble_step(T_global[gdofs], t))

        if pool is None or self._n_workers == 1:
            parts = [_one(eid) for eid in self._other_eids]
        else:
            parts = list(pool.map(_one, self._other_eids))
        Ks = [p[0] for p in parts]
        Ms = [p[1] for p in parts]

        # ── K ────────────────────────────────────────────────────────────────
        if not raw:
            sig = self._structure_signature(Ks)
            if sig is None or sig != self._pat_key.get("K_sig"):
                self._pat.pop("K", None)
                self._pat_key["K_sig"] = sig if sig is not None else ()
                if sig is None:
                    Ks = [self._to_sparse_K(K) for K in Ks]
            vals = [K.data for K in Ks]
        else:
            vals = list(Ks)
        if "K" not in self._pat:
            self._build_K_pattern_all(Ks if not raw else None)
        pat = self._pat["K"]
        K_data = self._K_const.copy()                   # joint links (constant)
        if vals:
            K_data += np.bincount(self._inv_other, weights=np.concatenate(vals),
                                  minlength=len(K_data))
        if self._batch is not None:
            Kb, Mb, Qb = self._batch.assemble(T_global, t)
            K_data += Kb
        K_global = pat.wrap(K_data)

        # ── Q ────────────────────────────────────────────────────────────────
        Q_out = (Qb.copy() if self._batch is not None else np.zeros(n))
        if parts:
            Q_out += np.bincount(self._gd_other, weights=np.concatenate([p[2] for p in parts]),
                                 minlength=n)

        if self._rad_exchange is not None:
            Q_out += self._rad_exchange.load(T_global, t)

        # ── M: lumped (1-D) → diagonal; any consistent member → triplet scatter ─
        diag = Mb.copy() if self._batch is not None else np.zeros(n)
        if all(not sp.issparse(M) and np.asarray(M).ndim == 1 for M in Ms):
            if Ms:
                diag += np.bincount(self._gd_other, weights=np.concatenate(Ms), minlength=n)
            M_global = sp.diags(diag, format="csr")
        else:
            M_gi, M_gj, M_val = zip(*(self._coo_triplets(M_e, self._gdof_map[eid])
                                      for eid, M_e in zip(self._other_eids, Ms)))
            M_global = (self._build("M", np.concatenate(M_gi), np.concatenate(M_gj),
                                    np.concatenate(M_val))
                        + sp.diags(diag, format="csr")).tocsr()
        return K_global, M_global, Q_out

    def _build_K_pattern_all(self, Ks: list | None) -> None:
        """Global K pattern: batch block, other members (raw structure or CSR), links."""
        from fahts.core.heat.solver.solid_solver import _CSRPattern

        gi, gj = [], []
        if self._batch is not None:
            bi, bj = self._batch.k_coo()
            gi.append(bi)
            gj.append(bj)
        for k, eid in enumerate(self._other_eids):
            gd = np.asarray(self._gdof_map[eid], dtype=np.intp)
            if Ks is None:
                indptr, indices = self._solvers[eid].k_structure()
                gi.append(gd[np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))])
                gj.append(gd[indices])
            else:
                ki, kj, _ = self._coo_triplets(Ks[k], gd)
                gi.append(ki)
                gj.append(kj)
        if self._k_link is not None:
            gi.append(self._k_link_coo.row)
            gj.append(self._k_link_coo.col)
        pat = _CSRPattern(np.concatenate(gi), np.concatenate(gj), self._n_total)
        self._pat["K"] = pat
        # split the pattern's inverse map: [batch | other members | joint links]
        inv = pat._inv
        n_b = len(self._batch.k_coo()[0]) if self._batch is not None else 0
        n_l = len(self._k_link_coo.data) if self._k_link is not None else 0
        nnz = pat._nnz
        if self._batch is not None:
            self._batch.bind(inv[:n_b], nnz)
        self._inv_other = inv[n_b:len(inv) - n_l]
        self._K_const = (np.bincount(inv[len(inv) - n_l:], weights=self._k_link_coo.data,
                                     minlength=nnz) if n_l else np.zeros(nnz))

    @staticmethod
    def _structure_signature(Ks: list) -> tuple | None:
        """Cheap identity of the member K structures (None → not all CSR)."""
        if not all(sp.issparse(K) and K.format == "csr" for K in Ks):
            return None
        return tuple(K.nnz for K in Ks)

    def _solve(self, A: sp.csr_matrix, b: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
        """Solve A x = b with the configured linear solver (A is SPD for CN)."""
        return self._spd.solve(A, b, x0)

    @property
    def cg_fallbacks(self) -> int:
        """Number of CG solves that fell back to the direct solver."""
        return self._spd.fallbacks

    def prescribed_values(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(global_dofs, temperatures) of all prescribed DOFs at time *t*.

        When several BCs pin the same DOF (e.g. a merged joint node) the last wins.
        """
        if not self._presc:
            return np.zeros(0, dtype=np.intp), np.zeros(0)
        dofs = np.concatenate([d for d, _ in self._presc])
        vals = np.concatenate([np.full(len(d), float(bc.eval(t))) for d, bc in self._presc])
        return dofs, vals

    def apply_prescribed(self, T_global: np.ndarray, t: float) -> None:
        """Overwrite prescribed DOFs of *T_global* in place with their values at *t*."""
        dofs, vals = self.prescribed_values(t)
        T_global[dofs] = vals

    def _solve_constrained(
        self, A: sp.csr_matrix, b: np.ndarray, dT_p: np.ndarray, x0: np.ndarray | None
    ) -> np.ndarray:
        """Solve A·x = b with x[pinned] = dT_p by symmetric elimination."""
        return self._spd.solve_constrained(A, b, self._free, self._pinned, dT_p, x0)

    def _init_rate(self, T_global: np.ndarray, pool) -> None:
        """Compute initial CN rate: Ṫ0 = M0⁻¹ · (Q0 − K0 · T0); pinned DOFs get Ṫ = 0."""
        K0, M0, Q0       = self._assemble_global(T_global, 0.0, pool)
        rhs = Q0 - K0 @ T_global
        n0 = M0.shape[0]
        f = self._free
        T_dot = np.zeros(n0)
        if (M0.nnz == n0 and np.array_equal(M0.indptr, np.arange(n0 + 1))
                and np.array_equal(M0.indices, np.arange(n0))):
            T_dot[f] = rhs[f] / M0.diagonal()[f]        # diagonal (lumped) mass
        else:
            # Free block only: consistent-mass coupling must not leak pinned-row residuals
            T_dot[f] = spla.spsolve(M0.tocsr()[f][:, f].tocsc(), rhs[f])
        self._T_dot_prev = T_dot
        self._K_prev     = K0
        self._M_prev     = M0

    def step(self, T_global: np.ndarray, dt: float, t: float) -> np.ndarray:
        """
        One Crank-Nicolson step ending at time *t*.

        Returns updated global temperature vector (n_total,).
        """
        two_over_dt = 2.0 / dt

        with ThreadPoolExecutor(max_workers=self._n_workers) if self._n_workers > 1 \
                else _NullPool() as pool:
            if self._K_prev is None:
                self._init_rate(T_global, pool)

            assert self._K_prev is not None
            assert self._M_prev is not None
            assert self._T_dot_prev is not None

            p_dofs, p_vals = self.prescribed_values(t)
            dT_p = np.zeros(len(self._pinned))
            if len(p_dofs):
                target = T_global.copy()
                target[p_dofs] = p_vals
                dT_p = target[self._pinned] - T_global[self._pinned]

            T_iter = T_global + dt * self._T_dot_prev
            T_iter[self._pinned] = T_global[self._pinned] + dT_p
            dT: np.ndarray | None = None
            K_i = M_i = None

            for _ in range(self._max_iter):
                K_i, M_i, Q_i = self._assemble_global(T_iter, t, pool)
                A  = (K_i + two_over_dt * M_i).tocsr()
                # Current-iterate matrices on the history side keep CN 2nd-order when
                # k(T)/c(T) vary (K_prev/M_prev here degraded it to 1st order).
                B  = (Q_i - K_i @ T_global
                      + M_i @ self._T_dot_prev)
                x0 = dT if dT is not None else dt * self._T_dot_prev
                dT    = self._solve_constrained(A, B, dT_p, x0)
                T_new = T_global + dT
                # Per-element convergence check (vectorised over the global DOF map):
                # max|ΔT_e| ≤ tol · max(1, max|T_e|) for every element e
                g = self._gd_concat
                d_e = np.maximum.reduceat(np.abs(T_new[g] - T_iter[g]), self._gd_offsets)
                s_e = np.maximum.reduceat(np.abs(T_new[g]), self._gd_offsets)
                if np.all(d_e <= self._tol * np.maximum(1.0, s_e)):
                    break
                T_iter = T_new

            assert dT is not None
            T_new     = T_global + dT
            T_dot_new = two_over_dt * dT - self._T_dot_prev
            T_dot_new[self._pinned] = 0.0

        # K_prev/M_prev only mark the CN state as initialised (the history terms use
        # the current iterate), so no extra end-of-step reassembly is needed.
        self._K_prev     = K_i
        self._M_prev     = M_i
        self._T_dot_prev = T_dot_new

        return T_new

    @property
    def gdof_map(self) -> dict[int, np.ndarray]:
        """Global DOF index array for each element's local nodes."""
        return self._gdof_map

    @property
    def n_total(self) -> int:
        """Total number of global thermal DOFs."""
        return self._n_total


def run_analysis(
    model: FEMModel,
    fire_zones: list,
    config: AnalysisConfig,
    progress_cb: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> TemperatureField:
    """
    Run the transient heat-transfer analysis for all exposed elements.

    Uses a time-step-outer / element-inner loop so that per-step global
    temperature statistics can be reported in real-time (matching USFOS output).
    Elements do not exchange heat with each other — each is driven solely by
    the fire boundary condition.

    Parameters
    ----------
    model:       Loaded FEMModel.
    fire_zones:  Active fire sources (FireZone and/or RadiationBall).
    config:      AnalysisConfig from RunAnalysisDialog.
    progress_cb: Optional callback(step, total_steps, -1) called each time step.
    cancel_check: Optional callable that returns True when cancelled.
    log_cb:      Optional callable(str) for USFOS-style console output.

    Returns
    -------
    TemperatureField with merged results for all analysed elements.

    Raises
    ------
    AnalysisCancelledError
        If cancel_check() returns True during the run.
    ValueError
        If no supported elements are exposed to any fire source.
    """
    config.validate()

    # ── Theory / engineering mode ─────────────────────────────────────────────
    # In theory mode the consistent mass matrix is enforced unconditionally.
    # We use a local variable so config is never mutated.
    if config.analysis_mode == "theory":
        effective_mass_matrix = "consistent"
        log.info(
            "Theory mode active: using consistent mass matrix "
            "(overrides config.mass_matrix=%r).",
            config.mass_matrix,
        )
    else:
        effective_mass_matrix = config.mass_matrix

    # Also accept ConcentratedSource / LineSource objects passed via config fields
    all_sources    = (
        list(fire_zones)
        + list(getattr(config, "concentrated_sources", []))
        + list(getattr(config, "line_sources", []))
    )
    active_sources = [s for s in all_sources if s.active]
    active_zones   = [s for s in active_sources if isinstance(s, FireZone)]
    active_balls   = [s for s in active_sources if isinstance(s, RadiationBall)]
    active_csrcs   = [s for s in active_sources if isinstance(s, ConcentratedSource)]
    active_lsrcs   = [s for s in active_sources if isinstance(s, LineSource)]

    use_3d = config.solver_dim == "3d"
    if use_3d:
        log.info("3-D solid solver: Hex8, %d layer(s) through thickness, %s linear solver.",
                 config.n_layers_3d, config.linear_solver)

    # ── Determine exposed elements ────────────────────────────────────────────
    if config.element_ids:
        beam_eids  = [e for e in config.element_ids if e in model.elements]
        shell_eids = [e for e in config.element_ids if e in model.shell_elements]
    else:
        beam_eids  = sorted(
            _exposed_beam_ids(model, active_zones, active_balls, active_csrcs, active_lsrcs)
        )
        shell_eids = sorted(
            _exposed_shell_ids(model, active_zones, active_balls, active_csrcs, active_lsrcs)
        )

    # model topology: free member ends / free plate edges are exposed in 3-D
    topo = _model_topology(model)

    # ── Build beam solvers ────────────────────────────────────────────────────
    solvers:       dict[int, object]         = {}
    meshes:        dict[int, object]         = {}
    T_states:      dict[int, np.ndarray]     = {}
    sec_type_name: dict[int, str]            = {}
    is_surface:    dict[int, bool]           = {}   # True → from_surface_solver_run

    valid_beam_eids: list[int] = []
    for eid in beam_eids:
        elem = model.elements[eid]
        sec  = model.sections.get(elem.geom_id)
        mat  = model.materials.get(elem.mat_id)
        if mat is not None and mat.rho == 0.0:
            # Rigid "dummy" material (MISOIEP rho=0) — USFOS still heats it using
            # the thermpar steel density.  Substitute a thermal-density copy so the
            # lumped mass matrix is non-singular; never mutate the shared material.
            log.debug(
                "Element %d: rho=0 (rigid material %d) — using thermal-density fallback.",
                eid, mat.mid,
            )
            mat = dataclasses.replace(mat, rho=_THERMAL_DENSITY_FALLBACK)
        if config.usfos_benchmark_mode and mat is not None:
            # Benchmark comparison: use the USFOS thermpar/tempdepy thermal tables
            # for every analysed element (never mutate the shared model material).
            mat = dataclasses.replace(mat, usfos_mode=True)
        if not isinstance(sec, (BoxSection, ISection, PipeSection)):
            log.warning(
                "Element %d: section type %s not supported — skipped.",
                eid, type(sec).__name__ if sec else "None",
            )
            continue
        midpoint = elem.midpoint(model.nodes)
        extra_pts = [model.nodes[elem.n1].xyz, model.nodes[elem.n2].xyz]
        (fire_temp, epsilon_m, h_conv, q_fn, covering_zones,
         ref_zone, covering_ball) = _bc_for_element(
            midpoint, active_zones, active_balls, active_csrcs, config,
            extra_pts=extra_pts, active_lsrcs=active_lsrcs,
        )

        if fire_temp is None:
            log.warning("Element %d: no active source covers midpoint — skipped.", eid)
            continue

        if use_3d:
            built = _build_solid_beam_solver(
                eid, elem, sec, mat, model, config, fire_temp, epsilon_m, h_conv, q_fn,
                covering_zones, ref_zone, covering_ball, active_zones, active_csrcs,
                active_lsrcs, effective_mass_matrix, topo=topo,
            )
            if built is None:
                continue
            solvers[eid], meshes[eid] = built
            T_states[eid]      = np.full(meshes[eid].n_nodes, 20.0)
            sec_type_name[eid] = type(sec).__name__
            is_surface[eid]    = True
            valid_beam_eids.append(eid)
            continue

        mesh = _build_beam_surface_mesh(sec, elem.length, config)

        # Heat-transfer-element-level per-quad exposure flags for FireZone sources.
        # Each quad centroid is tested in global coords; quads outside get 0 flux.
        # When all quads are exposed (None returned) behaviour is unchanged.
        quad_exposure: np.ndarray | None = None
        if covering_zones and active_zones:
            flags = element_quad_exposure_flags(mesh, elem, model.nodes, active_zones)
            if not np.any(flags > 0.0):
                log.debug(
                    "Element %d: zone covers endpoint/midpoint but no quad centroids — skipped.",
                    eid,
                )
                continue
            if not np.all(flags == 1.0):
                quad_exposure = flags

        # §3.3.4 geometric view factors (only for FireZone sources)
        vf = eps_s = eps_f = None
        if covering_zones:
            vf    = _element_view_factors(mesh, elem, model.nodes, covering_zones)
            eps_s = 0.7
            eps_f = ref_zone.effective_epsilon_fire
        elif config.usfos_benchmark_mode and covering_ball is not None:
            # RadiationBall path (epsilon_m=0): override the solver's internal 0.7
            # re-radiation emissivity with the USFOS reference value (fahts.fem emiss).
            eps_s = _USFOS_BENCHMARK_EMISSIVITY

        # §3.4.1: I/H profiles are open sections — every meshed plate (top flange,
        # web, bottom flange) is physically exposed on BOTH sides (no interior
        # cavity to shield either face), unlike BOX/PIPE outer walls which have a
        # genuine single exposed face.  Directional sources must therefore check
        # both the mesh's stored normal AND its mirror image and use whichever
        # side actually faces the source — see the `double_sided` handling in
        # _rad_ball_per_quad_flux / _concentrated_source_per_quad_flux /
        # _line_source_per_quad_flux below.
        double_sided = isinstance(sec, ISection)

        # RadiationBall: exact point-to-sphere flux per quad (engulfed or
        # exterior cos(θ)/(d/R)² regime — see _rad_ball_per_quad_flux).
        q_per_quad: np.ndarray | None = None
        if covering_ball is not None:
            q_per_quad = _rad_ball_per_quad_flux(
                covering_ball, mesh, elem, model.nodes, double_sided=double_sided
            )
            if np.all(q_per_quad == 0.0):
                log.debug("Element %d: ball flux zero for all faces — skipped.", eid)
                continue

        # §3.5.4 ConcentratedSource per-face directional flux: q=E·cos(θ)/(4π·r²)
        # Added when no RadiationBall is actively prescribing uniform flux for this element.
        # (RadiationBall uses a uniform prescribed q_fn; the concentrated source's per-quad
        # cos(θ) contribution would be double-counting in that case.)
        # FireZone + ConcentratedSource can coexist: the CS flux is added via q_per_quad.
        # §3.5.4/§3.5.5 point & line sources — flux ∝ E(t), so each source is stored as
        # E_s(t) × unit-power pattern and re-evaluated every step (TimeVaryingFaceFlux).
        # Skipped when a RadiationBall covers the element (double-counting guard).
        # An element is kept when any face CAN be lit (pattern > 0), even if E(0) = 0.
        q_dir: np.ndarray | TimeVaryingFaceFlux | None = q_per_quad
        if (active_csrcs or active_lsrcs) and covering_ball is None:
            sched = TimeVaryingFaceFlux(mesh.n_quads)
            for src in active_csrcs:
                sched.add_source(src, lambda u: _concentrated_source_per_quad_flux(
                    [u], mesh, elem, model.nodes, double_sided=double_sided))
            for src in active_lsrcs:
                sched.add_source(src, lambda u: _line_source_per_quad_flux(
                    [u], mesh, elem, model.nodes, double_sided=double_sided))
            if sched.lit:
                q_dir = sched
            elif not covering_zones:
                log.debug("Element %d: point/line source flux zero on all faces — skipped.",
                          eid)
                continue

        M_extra = _compute_M_extra(sec, mesh, elem.length)
        solvers[eid]       = SurfaceTransientSolver(
            mesh=mesh, material=mat, fire_temp=fire_temp,
            epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
            epsilon_steel=eps_s, epsilon_fire=eps_f, view_factors=vf,
            q_per_quad=q_dir, quad_exposure=quad_exposure,
            M_extra=M_extra, mass_matrix=effective_mass_matrix,
            insulation=config.insulation,
            prescribed_node_bcs=config.prescribed_node_bcs or None,
            # §3.4.1: I/H profiles are open sections — both faces of every plate
            # are fire-exposed ("2 outsides").  BOX/PIPE have 1 outside (default).
            n_exposed_sides=2 if isinstance(sec, ISection) else 1,
        )
        meshes[eid]        = mesh
        T_states[eid]      = np.full(mesh.n_nodes, 20.0)
        sec_type_name[eid] = type(sec).__name__
        is_surface[eid]    = True
        valid_beam_eids.append(eid)

    # ── Build shell solvers ───────────────────────────────────────────────────
    valid_shell_eids: list[int] = []
    for eid in shell_eids:
        shell = model.shell_elements[eid]
        sec   = model.sections.get(shell.geom_id)
        if not isinstance(sec, PlateSection):
            log.warning("Shell %d: section not PlateSection — skipped.", eid)
            continue
        mat      = model.materials[shell.mat_id]
        if mat is not None and mat.rho == 0.0:
            # Rigid "dummy" material (MISOIEP rho=0) — substitute thermal density
            # so the lumped mass matrix is non-singular (see beam loop above).
            log.debug(
                "Shell %d: rho=0 (rigid material %d) — using thermal-density fallback.",
                eid, mat.mid,
            )
            mat = dataclasses.replace(mat, rho=_THERMAL_DENSITY_FALLBACK)
        if config.usfos_benchmark_mode and mat is not None:
            # Benchmark comparison: use the USFOS thermpar/tempdepy thermal tables
            # for every analysed element (never mutate the shared model material).
            mat = dataclasses.replace(mat, usfos_mode=True)
        midpoint = np.mean([model.nodes[nid].xyz for nid in shell.nodes], axis=0)
        (fire_temp, epsilon_m, h_conv, q_fn, _czones,
         _rzone, shell_covering_ball) = _bc_for_element(
            midpoint, active_zones, active_balls, active_csrcs, config,
            active_lsrcs=active_lsrcs,
        )

        # §3.5.5 ConcentratedSource/LineSource coverage when no zone/ball covers midpoint
        shell_has_csrc = bool(active_csrcs)
        shell_has_lsrc = bool(active_lsrcs)
        if fire_temp is None:
            if not shell_has_csrc and not shell_has_lsrc:
                log.warning("Shell %d: no active source covers midpoint — skipped.", eid)
                continue
            fire_temp  = lambda t: 20.0   # noqa: E731
            epsilon_m  = 0.0
            h_conv     = 0.0
            q_fn       = lambda t: 0.0    # noqa: E731

        if len(shell.nodes) == 4 and use_3d:
            corners = np.array([model.nodes[nid].xyz for nid in shell.nodes], dtype=float)
            built_sh = _build_solid_plate_solver(
                eid, sec, mat, corners, config, fire_temp, epsilon_m, h_conv, q_fn,
                shell_covering_ball, active_csrcs, active_lsrcs, effective_mass_matrix,
                topo=topo, shell_nodes=tuple(shell.nodes),
            )
            if built_sh is None:
                continue
            solver, mesh = built_sh
            is_surf = True
            stype   = "QUADSHEL"
        elif len(shell.nodes) == 4:
            corners = np.array([model.nodes[nid].xyz for nid in shell.nodes])
            mesh    = PlateSurfaceMesher(
                section=sec, corners=corners,
                mesh_12=config.mesh_12, mesh_14=config.mesh_14,
            ).build()
            # Per-quad directional flux for QUADSHEL (RadiationBall)
            q_per_quad_sh: np.ndarray | None = None
            # Shell normal from corner geometry (uniform across flat plate)
            c    = corners
            sn   = np.cross(c[1] - c[0], c[3] - c[0])
            sn_n = np.linalg.norm(sn)
            sn   = sn / sn_n if sn_n > 1e-9 else sn
            # §3.4.1: QUADSHEL is exposed on both faces (like I/H profile plates —
            # see the `double_sided` note on _rad_ball_per_quad_flux).  `sn` is only
            # ONE of the two possible flat-plate normals, so directional sources must
            # check both `sn` and `-sn` and keep whichever side is actually lit.
            if shell_covering_ball is not None:
                # Uniform value across all quads (shell is flat): engulfed (d<=radius)
                # gets flux flat, exterior gets flux*(radius/d)^2*cos(θ) — whichever
                # face (sn or -sn) actually faces the ball.
                q_val = max(
                    shell_covering_ball.incident_flux(midpoint, sn),
                    shell_covering_ball.incident_flux(midpoint, -sn),
                )
                if q_val == 0.0:
                    log.debug("Shell %d: ball flux zero (facing away) — skipped.", eid)
                    continue
                q_per_quad_sh = np.full(mesh.n_quads, q_val)
            # §3.5.4 ConcentratedSource per-quad flux for QUADSHEL
            if active_csrcs or active_lsrcs:
                # Quad centroids: mean of 4 corner positions per quad
                quad_centroids = np.array([
                    mesh.nodes[mesh.quads[q]].mean(axis=0)
                    for q in range(mesh.n_quads)
                ])
                # All quads share the same flat-plate normal
                quad_normals = np.tile(sn, (mesh.n_quads, 1))
                # Flux ∝ E(t): store unit-power patterns (max of both plate sides, per
                # source) and re-evaluate every step via TimeVaryingFaceFlux.
                sched_sh = TimeVaryingFaceFlux(mesh.n_quads, static=q_per_quad_sh)
                for src in list(active_csrcs) + list(active_lsrcs):
                    sched_sh.add_source(src, lambda u: np.maximum(
                        u.per_quad_flux(quad_centroids, quad_normals),
                        u.per_quad_flux(quad_centroids, -quad_normals)))
                if sched_sh.terms:
                    q_per_quad_sh = sched_sh
            # RadiationBall re-radiation emissivity override for benchmark mode.
            eps_s_sh = (
                _USFOS_BENCHMARK_EMISSIVITY
                if config.usfos_benchmark_mode and shell_covering_ball is not None
                else None
            )
            solver  = SurfaceTransientSolver(
                mesh=mesh, material=mat, fire_temp=fire_temp,
                epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
                q_per_quad=q_per_quad_sh, epsilon_steel=eps_s_sh,
                mass_matrix=effective_mass_matrix,
                # §3.4.1 — both outsides exposed; skip doubling for directional per-quad flux
                n_exposed_sides=1 if q_per_quad_sh is not None else 2,
                insulation=config.insulation,
                prescribed_node_bcs=config.prescribed_node_bcs or None,
            )
            is_surf = True
            stype   = "QUADSHEL"
        else:
            if shell_covering_ball is not None:
                log.debug("Shell %d: TRISHELL RadiationBall not supported — skipped.", eid)
                continue
            mesh    = ShellMesher(sec, n_layers=config.n_layers).build()
            solver  = Shell1DSolver(
                mesh=mesh, material=mat, fire_temp=fire_temp,
                epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
                # §3.4.1 — both outsides exposed; inner face sees same fire as outer
                fire_temp_inner=fire_temp,
                mass_matrix=effective_mass_matrix,
            )
            is_surf = False
            stype   = "TRISHELL"
        solvers[eid]       = solver
        meshes[eid]        = mesh
        T_states[eid]      = np.full(mesh.n_nodes, 20.0)
        sec_type_name[eid] = stype
        is_surface[eid]    = is_surf
        valid_shell_eids.append(eid)

    all_eids = valid_beam_eids + valid_shell_eids
    total    = len(all_eids)
    if total == 0:
        raise ValueError(
            "No supported elements are exposed to fire. "
            "Supported cross-sections: BOX, IHPROFIL, PIPE, QUADSHEL/TRISHELL."
        )

    # ── USFOS-style header + step-table column headers ────────────────────────
    n_steps   = max(1, round(config.t_end / config.dt))
    out_every = max(1, round(config.output_dt / config.dt))

    if log_cb is not None:
        _log_header(log_cb, model, config, len(valid_beam_eids), len(valid_shell_eids),
                    total, active_zones, active_balls, active_csrcs, active_lsrcs)

    # ── Output storage ────────────────────────────────────────────────────────
    out_times: list[float]                   = [0.0]
    out_T:     dict[int, list[np.ndarray]]   = {eid: [T_states[eid].copy()] for eid in all_eids}

    # ── Build global DOF map (co-located node merging at 1 mm tolerance) ─────
    # Single-threaded member assembly: per-member work is small numpy calls that hold
    # the GIL, so a thread pool was measured ~30 % SLOWER (model_file.fem, 2026-09-27).
    n_workers = 1
    _gpos     = _compute_global_node_positions(all_eids, meshes, model, is_surface)
    k_link: sp.csr_matrix | None = None
    rad_exchange = None
    if use_3d:
        # Safe proximity merge (never collapses a member) + joint conductance links
        gdof_map, n_global = _si.build_solid_dof_map(all_eids, meshes, _gpos, tol=1e-3)
        k_link = _si.build_joint_links(all_eids, meshes, _gpos, gdof_map, n_global, model)
        if config.shielding or config.radiation_exchange:
            rad_exchange = _setup_radiation_geometry(
                all_eids, meshes, _gpos, solvers, gdof_map, n_global, config, log_cb
            )
    else:
        gdof_map, n_global = _build_global_dof_map(
            all_eids, meshes, _gpos, tol=1e-3, model=model
        )
    n_raw    = sum(meshes[eid].n_nodes for eid in all_eids)
    n_merged = n_raw - n_global
    if n_merged > 0:
        log.info(
            "Node merging: %d co-located nodes merged → %d global DOFs (was %d).",
            n_merged, n_global, n_raw,
        )
    global_solver = GlobalThermalSolver(
        rad_exchange=rad_exchange,
        eids=all_eids,
        solvers=solvers,
        gdof_map=gdof_map,
        n_global_dofs=n_global,
        n_workers=n_workers,
        linear_solver=config.linear_solver,
        k_link=k_link,
    )
    T_global = np.full(n_global, 20.0)
    global_solver.apply_prescribed(T_global, 0.0)
    for eid in all_eids:
        T_states[eid] = T_global[global_solver.gdof_map[eid]]
        out_T[eid][0] = T_states[eid].copy()

    # ── Time-step outer loop ──────────────────────────────────────────────────
    t        = 0.0
    step_log = 0

    for step_i in range(n_steps):
        dt_step = min(config.dt, config.t_end - t)
        if dt_step <= 0.0:
            break
        t += dt_step

        if cancel_check is not None and cancel_check():
            raise AnalysisCancelledError(f"Analysis cancelled at t={t:.2f} s.")

        # One global CN step — coupled sparse system, parallel element assembly
        T_global = global_solver.step(T_global, dt_step, t)

        # Scatter global vector back to per-element views for logging / output
        for eid in all_eids:
            T_states[eid] = T_global[global_solver.gdof_map[eid]]

        # Emit one USFOS-style step row at every internal step
        step_log += 1
        if log_cb is not None:
            T_max_now = max(float(np.max(T_states[eid])) for eid in all_eids)
            T_min_now = min(float(np.min(T_states[eid])) for eid in all_eids)
            _log_step_row(log_cb, step_log, t / 60.0, T_max_now, T_min_now)

        # Store output at intervals
        if (step_i + 1) % out_every == 0 or step_i == n_steps - 1:
            out_times.append(t)
            for eid in all_eids:
                out_T[eid].append(T_states[eid].copy())

        if progress_cb is not None:
            progress_cb(step_i + 1, n_steps, -1)

    if progress_cb is not None:
        progress_cb(n_steps, n_steps, -1)

    # ── Build TemperatureField per element ────────────────────────────────────
    times_arr = np.array(out_times)
    fields: list[TemperatureField] = []

    if log_cb is not None:
        _log_element_section_header(log_cb)

    for eid in valid_beam_eids:
        T_hist = np.array(out_T[eid])
        elem   = model.elements[eid]
        R_beam = _beam_local_to_global(elem)
        origin = model.nodes[elem.n1].xyz
        if elem.ecc1 is not None:
            origin = origin + np.asarray(elem.ecc1, dtype=float)
        mesh_e = meshes[eid]
        nodes_global = origin + (R_beam @ mesh_e.nodes.T).T   # beam-local → global
        if isinstance(mesh_e, SolidMesh):
            tf = TemperatureField.from_solid_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh_e,
                nodes_global=nodes_global,
            )
        else:
            tf = TemperatureField.from_surface_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh_e,
                nodes_global=nodes_global,
            )
        fields.append(tf)
        t_peak = tf.peak_centroid_temperature(eid)
        log.info("Beam %d (%s) solved: T_peak=%.1f °C", eid, sec_type_name[eid], t_peak)
        if log_cb is not None:
            _log_element_row(log_cb, eid, sec_type_name[eid], t_peak)

    for eid in valid_shell_eids:
        T_hist = np.array(out_T[eid])
        mesh   = meshes[eid]
        if isinstance(mesh, SolidMesh):
            # PlateSolidMesher nodes are global
            tf = TemperatureField.from_solid_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh,
                nodes_global=np.asarray(mesh.nodes, dtype=float).copy(),
            )
        elif is_surface[eid]:
            # PlateSurfaceMesher nodes are built from global corner coords → already global
            tf = TemperatureField.from_surface_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh,
                nodes_global=mesh.nodes.copy(),
            )
        else:
            tf = TemperatureField.from_shell_solver_run(
                eid=eid, times=times_arr, T_history=T_hist, mesh=mesh
            )
        fields.append(tf)
        t_peak = tf.peak_centroid_temperature(eid)
        log.info("Shell %d solved: T_peak=%.1f °C", eid, t_peak)
        if log_cb is not None:
            _log_element_row(log_cb, eid, sec_type_name[eid], t_peak)

    if not fields:
        raise ValueError("Analysis produced no results — all elements were skipped.")

    merged = TemperatureField.merge(fields)
    log.info(
        "Analysis complete: %d elements solved, %d time steps.",
        merged.n_elements, merged.n_steps,
    )
    if log_cb is not None:
        _log_summary(log_cb, merged)

    return merged


# ── 3-D solid solver construction ─────────────────────────────────────────────

def _solid_eps_rerad(config: AnalysisConfig, covering_ball) -> float | None:
    """Re-radiation emissivity override for the RadiationBall path (benchmark mode)."""
    if config.usfos_benchmark_mode and covering_ball is not None:
        return _USFOS_BENCHMARK_EMISSIVITY
    return None


def _model_topology(model: FEMModel) -> tuple[dict[int, int], dict[frozenset, int]]:
    """
    (node use count, edge use count) over all beams and shells.  A beam counts as the
    edge between its two nodes, so a plate edge running along a beam is not "free".
    """
    node_use: dict[int, int] = {}
    edge_use: dict[frozenset, int] = {}
    for el in model.elements.values():
        for n in (el.n1, el.n2):
            node_use[n] = node_use.get(n, 0) + 1
        k = frozenset((el.n1, el.n2))
        edge_use[k] = edge_use.get(k, 0) + 1
    for sh in getattr(model, "shell_elements", {}).values():
        ns = list(sh.nodes)
        for n in ns:
            node_use[n] = node_use.get(n, 0) + 1
        for a, b in zip(ns, ns[1:] + ns[:1]):
            k = frozenset((a, b))
            edge_use[k] = edge_use.get(k, 0) + 1
    return node_use, edge_use


def _expose_free_beam_ends(mesh: SolidMesh, elem, node_use: dict[int, int]) -> None:
    """End caps at member ends connected to nothing else → FACE_OUTER (in place)."""
    from fahts.core.heat.solid_mesh import FACE_END, FACE_OUTER

    fe = mesh.face_indices(FACE_END)
    if len(fe) == 0:
        return
    x = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.faces)[fe], 0]    # (n_fe, 4)
    tol = 1e-9 * max(1.0, elem.length)
    for nid, x_end in ((elem.n1, 0.0), (elem.n2, elem.length)):
        if node_use.get(nid, 0) <= 1:
            sel = fe[np.all(np.abs(x - x_end) <= tol, axis=1)]
            mesh.face_group[sel] = FACE_OUTER
    mesh._cache.clear()


def _build_solid_beam_solver(
    eid: int,
    elem,
    sec,
    mat,
    model: FEMModel,
    config: AnalysisConfig,
    fire_temp,
    epsilon_m: float,
    h_conv: float,
    q_fn,
    covering_zones: list,
    ref_zone,
    covering_ball,
    active_zones: list,
    active_csrcs: list,
    active_lsrcs: list,
    mass_matrix: str,
    topo: tuple | None = None,
) -> tuple[SolidTransientSolver, SolidMesh] | None:
    """
    Build the Hex8 SolidMesh + SolidTransientSolver of one beam member (3-D path).

    Per-face boundary data are evaluated on FACE_OUTER faces in GLOBAL coordinates
    (R = _beam_local_to_global(elem), origin = n1 + ecc1).  Every outer face has a
    true outward normal, so I-profile flanges/web are exposed on both faces without
    any double-sided correction.  Returns None when the member receives no heat
    (same skip rules as the 2-D path).
    """
    mesh = _si.build_beam_solid_mesh(sec, elem.length, config)
    if topo is not None:
        # free member ends (connected to nothing) expose their end caps to the fire
        _expose_free_beam_ends(mesh, elem, topo[0])
    R = _beam_local_to_global(elem)
    origin = np.asarray(model.nodes[elem.n1].xyz, dtype=float).copy()
    if elem.ecc1 is not None:
        origin = origin + np.asarray(elem.ecc1, dtype=float)
    corners, cents, normals = _si.outer_face_geometry(mesh, R, origin)

    face_exposure: np.ndarray | None = None
    if covering_zones and active_zones:
        flags = _si.zone_face_exposure(cents, active_zones)
        if not np.any(flags > 0.0):
            log.debug("Element %d: zone covers endpoint/midpoint but no faces — skipped.", eid)
            return None
        if not np.all(flags == 1.0):
            face_exposure = flags

    vf = eps_s = eps_f = None
    if covering_zones:
        vf    = _si.face_view_factors(corners, normals, covering_zones)
        eps_s = 0.7
        eps_f = ref_zone.effective_epsilon_fire
    else:
        eps_s = _solid_eps_rerad(config, covering_ball)

    q_face: np.ndarray | None = None
    if covering_ball is not None:
        q_face = _si.ball_face_flux(covering_ball, cents, normals)
        # In 3-D an element with no DIRECT flux is kept: it is heated by conduction from
        # connected lit elements and by radiation exchange (the 2-D solver skipped it).
    q_dir: np.ndarray | TimeVaryingFaceFlux | None = q_face
    if covering_ball is None and (active_csrcs or active_lsrcs):
        # Flux ∝ E(t): unit-power pattern per source, re-evaluated each step.
        sched = TimeVaryingFaceFlux(len(cents))
        for src in list(active_csrcs) + list(active_lsrcs):
            sched.add_source(src, lambda u: u.per_quad_flux(cents, normals))
        if sched.lit:
            q_dir = sched
        # (no skip when unlit — see the RadiationBall note above)

    M_extra = _si.solid_M_extra(sec, mesh, elem.length, config.enclosed_gas_rho_c)
    solver = SolidTransientSolver(
        mesh=mesh, material=mat, fire_temp=fire_temp,
        epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
        epsilon_steel=eps_s, epsilon_fire=eps_f, view_factors=vf,
        q_per_face=q_dir, face_exposure=face_exposure,
        M_extra=M_extra, mass_matrix=mass_matrix,
        insulation=config.insulation,
        prescribed_node_bcs=config.prescribed_node_bcs or None,
        conduction=config.conduction_3d,
    )
    solver._shield_ball = covering_ball          # for source shielding (radiation/)
    return solver, mesh


def _build_solid_plate_solver(
    eid: int,
    sec: PlateSection,
    mat,
    corners: np.ndarray,
    config: AnalysisConfig,
    fire_temp,
    epsilon_m: float,
    h_conv: float,
    q_fn,
    covering_ball,
    active_csrcs: list,
    active_lsrcs: list,
    mass_matrix: str,
    topo: tuple | None = None,
    shell_nodes: tuple | None = None,
) -> tuple[SolidTransientSolver, SolidMesh] | None:
    """
    Build the Hex8 PlateSolidMesher mesh + solver of one QUADSHEL (3-D path).

    Both plate faces are FACE_OUTER with true opposite normals, so a fire zone heats
    both sides and a directional source lights only the side facing it.
    """
    exposed = (False, False, False, False)
    if topo is not None and shell_nodes is not None:
        # free plate edges (not shared with another shell or a beam) are exposed:
        # their thickness faces receive fire / source heat
        ns = list(shell_nodes)
        exposed = tuple(topo[1].get(frozenset((a, b)), 0) <= 1
                        for a, b in zip(ns, ns[1:] + ns[:1]))
    mesh = PlateSolidMesher(
        section=sec, corners=corners, mesh_12=config.mesh_12, mesh_14=config.mesh_14,
        n_layers=config.n_layers_3d, exposed_edges=exposed,
    ).build()
    corners_f, cents, normals = _si.outer_face_geometry(mesh)
    q_face: np.ndarray | None = None
    if covering_ball is not None:
        q_face = _si.ball_face_flux(covering_ball, cents, normals)
        # kept even when unlit: heated by conduction / radiation exchange (3-D)
    elif active_csrcs or active_lsrcs:
        # Flux ∝ E(t): unit-power pattern per source, re-evaluated each step.
        sched = TimeVaryingFaceFlux(len(cents))
        for src in list(active_csrcs) + list(active_lsrcs):
            sched.add_source(src, lambda u: u.per_quad_flux(cents, normals))
        if sched.lit:
            q_face = sched
    solver = SolidTransientSolver(
        mesh=mesh, material=mat, fire_temp=fire_temp,
        epsilon_m=epsilon_m, h_conv=h_conv, T0=20.0, q_prescribed_fn=q_fn,
        q_per_face=q_face, epsilon_steel=_solid_eps_rerad(config, covering_ball),
        mass_matrix=mass_matrix,
        insulation=config.insulation,
        prescribed_node_bcs=config.prescribed_node_bcs or None,
        conduction=config.conduction_3d,
    )
    solver._shield_ball = covering_ball
    return solver, mesh


# ── Exposure helpers ──────────────────────────────────────────────────────────

def _exposed_beam_ids(
    model,
    active_zones,
    active_balls,
    active_csrcs: list[ConcentratedSource] | None = None,
    active_lsrcs: list[LineSource] | None = None,
) -> set[int]:
    result: set[int] = set()
    if active_zones:
        result |= exposed_element_ids(model.elements, active_zones, model.nodes)
    for ball in active_balls:
        result |= ball.exposed_element_ids(
            model.elements, model.nodes, min_flux=_MIN_BALL_FLUX
        ).keys()
    # §3.5.4 ConcentratedSource: every structural element is potentially exposed
    # (the source radiates in all directions with unlimited range).  Include all
    # beam elements with any facing quads; exact per-quad clipping is done later
    # when q_per_quad is assembled.  Here we conservatively include all elements.
    if active_csrcs:
        result |= set(model.elements.keys())
    # §3.5.5 LineSource: same conservative approach as ConcentratedSource — the
    # line radiates in all directions with unlimited range; exact per-quad flux
    # clipping (cos(θ) ≤ 0 → 0) is applied later in _line_source_per_quad_flux.
    if active_lsrcs:
        result |= set(model.elements.keys())
    return result


def _setup_radiation_geometry(eids, meshes, positions, solvers, gdof_map, n_global,
                              config: AnalysisConfig, log_cb=None):
    """
    3-D radiation geometry: build one triangle scene of all member surfaces, shield the
    directional sources (in place on the member solvers) and build the surface-to-surface
    exchange.  Returns the RadiationExchange (or None).
    """
    import time as _time

    from fahts.core.heat.radiation.exchange import RadiationExchange
    from fahts.core.heat.radiation.shielding import apply_source_shielding, build_scene

    t0 = _time.perf_counter()
    scene, members = build_scene(eids, meshes, positions)
    if scene is None:
        return None
    if config.shielding:
        balls = {e: getattr(solvers[e], "_shield_ball", None) for e in members}
        stats = apply_source_shielding(scene, members, solvers, balls)
        msg = (f"Source shielding: {stats['lit_faces']:.0f} lit faces, "
               f"{stats['shaded_face_equiv']:.1f} face-equivalents shaded.")
        log.info(msg)
        if log_cb is not None:
            log_cb(" " + msg)
    exchange = None
    if config.radiation_exchange:
        exchange = RadiationExchange(scene, members, solvers, gdof_map, n_global,
                                     patch_size=config.rad_patch_size,
                                     rays=config.rad_rays_per_patch,
                                     balls={e: getattr(solvers[e], "_shield_ball", None)
                                            for e in members})
        if log_cb is not None:
            log_cb(f" Radiation exchange: {exchange.n_patches} patches.")
    log.info("Radiation geometry set-up: %.2f s", _time.perf_counter() - t0)
    return exchange


def exposed_analysis_element_ids(model: FEMModel, sources: list) -> set[int]:
    """
    IDs of beam AND shell elements that at least one ACTIVE source exposes, using the
    same screening as ``run_analysis`` (FireZone, RadiationBall, ConcentratedSource,
    LineSource).  Used by the GUI to count exposed elements before a run.
    """
    active = [s for s in sources if getattr(s, "active", True)]
    zones = [s for s in active if isinstance(s, FireZone)]
    balls = [s for s in active if isinstance(s, RadiationBall)]
    csrcs = [s for s in active if isinstance(s, ConcentratedSource)]
    lsrcs = [s for s in active if isinstance(s, LineSource)]
    return (set(_exposed_beam_ids(model, zones, balls, csrcs, lsrcs))
            | _exposed_shell_ids(model, zones, balls, csrcs, lsrcs))


def _exposed_shell_ids(
    model,
    active_zones,
    active_balls,
    active_csrcs: list[ConcentratedSource] | None = None,
    active_lsrcs: list[LineSource] | None = None,
) -> set[int]:
    if not model.shell_elements:
        return set()
    result: set[int] = set()
    for shell in model.shell_elements.values():
        mid = np.mean([model.nodes[nid].xyz for nid in shell.nodes], axis=0)
        for zone in active_zones:
            if zone.contains_midpoint(mid):
                result.add(shell.eid)
                break
        else:
            for ball in active_balls:
                dist = float(np.linalg.norm(mid - ball.center))
                if ball.max_flux_at_distance(dist) > _MIN_BALL_FLUX:
                    result.add(shell.eid)
                    break
            else:
                # §3.5.4 ConcentratedSource / §3.5.5 LineSource: include shells
                # (exact clipping done later by per-quad flux computation)
                if active_csrcs or active_lsrcs:
                    result.add(shell.eid)
    return result


def _bc_for_element(
    midpoint: np.ndarray,
    active_zones: list[FireZone],
    active_balls: list[RadiationBall],
    active_csrcs: list[ConcentratedSource],
    config: AnalysisConfig,
    extra_pts: list[np.ndarray] | None = None,
    active_lsrcs: list[LineSource] | None = None,
) -> tuple:
    """
    Determine fire BC parameters for an element.

    Checks `midpoint` plus any additional `extra_pts` (e.g. beam endpoints) so
    that elements straddling a zone boundary are correctly identified.

    Returns (fire_temp_fn, epsilon_m, h_conv, q_prescribed_fn,
             covering_zones, ref_zone, covering_ball).

    ``covering_ball`` is the RadiationBall giving the strongest flux (direction-
    agnostic upper bound) at the midpoint (or None), gated on `_MIN_BALL_FLUX`.
    When set, the caller builds a per-quad flux array via
    ``_rad_ball_per_quad_flux`` rather than applying a uniform flux to every
    face. In that case ``q_prescribed_fn`` returns 0.

    Returns (None, ...) when no source covers any of the check points.

    Priority:
        RadiationBall > ConcentratedSource/LineSource > FireZone when sources overlap.
        RadiationBall, ConcentratedSource, and LineSource all set epsilon_m=0 (flux
        is fully prescribed per-quad; re-radiation is handled by eps_rerad).
    """
    covering_balls: list[tuple[RadiationBall, float]] = []
    for b in active_balls:
        dist = float(np.linalg.norm(midpoint - b.center))
        flux = b.max_flux_at_distance(dist)
        if flux > _MIN_BALL_FLUX:
            covering_balls.append((b, flux))

    if covering_balls:
        ref_ball, _ref_flux = max(covering_balls, key=lambda x: x[1])
        fire_temp = lambda t: 20.0      # noqa: E731
        # Flux is applied per-quad with cos(θ) directionality by the caller, so
        # the uniform q_fn contributes nothing here.
        q_fn = lambda _t: 0.0           # noqa: E731
        return fire_temp, 0.0, 0.0, q_fn, [], None, ref_ball

    # Collect zones covering any of the check points (midpoint + endpoints)
    check_pts = [midpoint] + (extra_pts or [])
    covering_set: set[int] = set()
    covering_zones: list[FireZone] = []
    for pt in check_pts:
        for z in active_zones:
            if id(z) not in covering_set and z.contains_point(pt):
                covering_set.add(id(z))
                covering_zones.append(z)

    if not covering_zones:
        # ConcentratedSource/LineSource: unlimited range — every element is potentially
        # covered.  The actual per-quad flux is applied via q_per_quad in the solver;
        # here we only need to mark the element as active (fire_temp=ambient, eps=0, h=0).
        if active_csrcs or active_lsrcs:
            fire_temp = lambda t: 20.0   # noqa: E731
            return fire_temp, 0.0, 0.0, lambda _t: 0.0, [], None, None
        return None, 0.0, 0.0, lambda _t: 0.0, [], None, None

    if len(covering_zones) == 1:
        fire_temp = covering_zones[0].temperature
    else:
        def fire_temp(t, _zones=covering_zones):
            return max(z.temperature(t) for z in _zones)

    ref_zone  = max(covering_zones, key=lambda z: z.temperature(config.t_end))
    epsilon_m = ref_zone.effective_epsilon_fire * 0.7
    h_conv    = ref_zone.h_conv
    return fire_temp, epsilon_m, h_conv, lambda _t: 0.0, covering_zones, ref_zone, None


def _rad_ball_per_quad_flux(
    ball: RadiationBall,
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
    double_sided: bool = False,
) -> np.ndarray:
    """
    Compute per-quad prescribed flux [W/m²] for a beam element covered by a ball.

    Evaluated independently at each quad's own centroid via
    ``ball.incident_flux(centroid, normal)`` (see rad_ball.py):
        d <= radius  : q = ball.flux                              (engulfed — all faces)
        d >  radius  : q = ball.flux * (radius/d)**2 * cos(θ)     (exterior, 0 if facing away)

    where d is the ball-centre-to-quad-centroid distance and θ is the angle
    between the face outward normal and the direction from that specific
    quad centroid toward the ball centre.

    ``double_sided=True`` (I/H profiles — see §3.4.1 note at the call site):
    the mesh stores only ONE fixed normal per plate (e.g. the I-beam web is
    only ever meshed on one lateral side), but the plate is physically exposed
    on both faces.  For any single external point source, at most one of a
    flat plate's two opposite normals can have cos(θ) > 0 — so this evaluates
    both ``normal`` and ``-normal`` and keeps whichever is actually lit,
    instead of silently zeroing the quad when the ball happens to be on the
    side the mesh's stored normal doesn't point toward.  This never sums both
    sides (that would double-count) — exactly one is ever nonzero for a
    single point source.

    Parameters
    ----------
    ball         : RadiationBall source
    mesh         : BeamSurfaceMesh for this element (beam-local coords)
    elem         : BeamElement (provides direction, local_z, n1 for frame)
    model_nodes  : {nid: Node} global node positions
    double_sided : True for open-profile sections (I/H) whose plates are
                   exposed on both faces; False for BOX/PIPE outer walls
                   (single genuine exposed face — the other side faces a
                   sealed interior cavity and must NOT pick up ball flux).

    Returns
    -------
    (n_quads,) array of per-quad flux values [W/m²]
    """
    R      = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    q_per_quad = np.zeros(mesh.n_quads)
    for q in range(mesh.n_quads):
        # Quad centroid in global coordinates
        centroid_local  = mesh.nodes[mesh.quads[q]].mean(axis=0)   # (3,) beam-local
        centroid_global = origin + R @ centroid_local               # global

        normal_local  = _quad_outward_normal_local(mesh, q)
        normal_global = R @ normal_local
        norm = np.linalg.norm(normal_global)
        if norm > 1e-9:
            normal_global = normal_global / norm

        q_val = ball.incident_flux(centroid_global, normal_global)
        if double_sided:
            q_val = max(q_val, ball.incident_flux(centroid_global, -normal_global))
        q_per_quad[q] = q_val

    return q_per_quad


def _concentrated_source_per_quad_flux(
    csrcs: list[ConcentratedSource],
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
    t: float = 0.0,
    double_sided: bool = False,
) -> np.ndarray:
    """
    Compute per-quad prescribed flux [W/m²] from all ConcentratedSource objects.

    FAHTS §3.5.4:
        q_i = E(t) * cos(theta_i) / (4 * pi * r_i^2)

    where theta_i is the angle between the quad outward normal and the direction
    from the source to the quad centroid.  Faces pointing away receive 0.

    Per-quad centroids are computed in global coordinates by transforming
    beam-local quad node positions using the beam frame rotation matrix.

    ``double_sided=True``: for each source independently, evaluate both
    ``quad_normals`` and ``-quad_normals`` and keep whichever side that
    source actually lights up per quad (see _rad_ball_per_quad_flux for the
    full rationale — I/H profile plates are exposed on both faces).  Done
    per-source-then-summed (not summed-then-maxed) so multiple sources
    hitting opposite faces of the same plate both contribute correctly.

    Parameters
    ----------
    csrcs        : list of active ConcentratedSource objects
    mesh         : BeamSurfaceMesh for this element (beam-local coords)
    elem         : BeamElement (provides direction, local_z, n1 for frame)
    model_nodes  : {nid: Node} global node positions
    t            : simulation time [s] (for time-dependent power)
    double_sided : True for open-profile (I/H) plates exposed on both faces.

    Returns
    -------
    (n_quads,) array of summed per-quad flux values [W/m²]
    """
    R      = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    # Compute global centroid and outward normal for each quad
    quad_centroids = np.zeros((mesh.n_quads, 3))
    quad_normals   = np.zeros((mesh.n_quads, 3))
    for q in range(mesh.n_quads):
        node_q_local     = mesh.nodes[mesh.quads[q]]           # (4, 3) beam-local
        centroid_local   = node_q_local.mean(axis=0)           # (3,)
        quad_centroids[q] = origin + R @ centroid_local        # global
        normal_local     = _quad_outward_normal_local(mesh, q)
        normal_global    = R @ normal_local
        norm             = np.linalg.norm(normal_global)
        quad_normals[q]  = normal_global / norm if norm > 1e-9 else normal_global

    q_total = np.zeros(mesh.n_quads)
    for csrc in csrcs:
        q_pos = csrc.per_quad_flux(quad_centroids, quad_normals, t=t)
        if double_sided:
            q_neg = csrc.per_quad_flux(quad_centroids, -quad_normals, t=t)
            q_total += np.maximum(q_pos, q_neg)
        else:
            q_total += q_pos
    return q_total


def _line_source_per_quad_flux(
    lsrcs: list[LineSource],
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
    t: float = 0.0,
    n_segments: int = 10,
    double_sided: bool = False,
) -> np.ndarray:
    """
    Compute per-quad prescribed flux [W/m²] from all LineSource objects.

    FAHTS §3.5.5: the line is divided into n_segments discrete sub-sources each
    treated as a concentrated source (§3.5.4).  End sub-sources emit 50% of the
    interior energy.  Contributions are summed across all sub-sources and sources.

    Per-quad centroids are computed in global coordinates by transforming
    beam-local quad node positions using the beam frame rotation matrix.

    ``double_sided=True``: see _concentrated_source_per_quad_flux — evaluated
    per-source-then-summed so multiple sources on opposite faces of the same
    plate both contribute correctly.

    Parameters
    ----------
    lsrcs        : list of active LineSource objects
    mesh         : BeamSurfaceMesh for this element (beam-local coords)
    elem         : BeamElement (provides direction, local_z, n1 for frame)
    model_nodes  : {nid: Node} global node positions
    t            : simulation time [s] (for time-dependent power)
    n_segments   : discrete sub-sources per line source (default 10)
    double_sided : True for open-profile (I/H) plates exposed on both faces.

    Returns
    -------
    (n_quads,) array of summed per-quad flux values [W/m²]
    """
    R      = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    quad_centroids = np.zeros((mesh.n_quads, 3))
    quad_normals   = np.zeros((mesh.n_quads, 3))
    for q in range(mesh.n_quads):
        node_q_local      = mesh.nodes[mesh.quads[q]]
        centroid_local    = node_q_local.mean(axis=0)
        quad_centroids[q] = origin + R @ centroid_local
        normal_local      = _quad_outward_normal_local(mesh, q)
        normal_global     = R @ normal_local
        norm              = np.linalg.norm(normal_global)
        quad_normals[q]   = normal_global / norm if norm > 1e-9 else normal_global

    q_total = np.zeros(mesh.n_quads)
    for lsrc in lsrcs:
        q_pos = lsrc.per_quad_flux(quad_centroids, quad_normals, t=t, n_segments=n_segments)
        if double_sided:
            q_neg = lsrc.per_quad_flux(quad_centroids, -quad_normals, t=t, n_segments=n_segments)
            q_total += np.maximum(q_pos, q_neg)
        else:
            q_total += q_pos
    return q_total


def _beam_local_to_global(elem) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (origin_global, R) for transforming beam-local → global coords.

    origin_global : global position of beam start node (ignoring eccentricity,
                    which is small relative to fire zone dimensions)
    R             : (3,3) rotation matrix; columns = [local_x, local_y, local_z]
    """
    local_x = np.asarray(elem.direction, dtype=float)
    local_z = np.asarray(elem.local_z, dtype=float)
    local_y = np.cross(local_x, local_z)
    norm_y = np.linalg.norm(local_y)
    local_y = local_y / norm_y if norm_y > 1e-9 else np.array([0.0, 1.0, 0.0])
    R = np.column_stack([local_x, local_y, local_z])
    return R


def _quad_outward_normal_local(mesh: BeamSurfaceMesh, q: int) -> np.ndarray:
    """
    Outward unit normal of quad q in beam-local coordinates.

    BOX/I sections: determined from local_cols (face-plane orientation + sign).
    PIPE sections : radial direction from the beam axis in the y-z plane.
    """
    node_q = mesh.nodes[mesh.quads[q]]          # (4, 3) beam-local
    if mesh.local_coords_2d is not None:
        # PIPE: outward = radial from beam axis
        c = node_q.mean(axis=0)
        radial = np.array([0.0, c[1], c[2]])
        norm = np.linalg.norm(radial)
        return radial / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    lc = mesh.local_cols[q]
    if lc[1] == 1:          # (0,1) → x-y plane, constant-z face (top/bottom)
        z_val = float(node_q[0, 2])
        return np.array([0.0, 0.0, np.sign(z_val)])
    else:                   # (0,2) → x-z plane, constant-y face (left/right)
        y_val = float(node_q[0, 1])
        return np.array([0.0, np.sign(y_val), 0.0])


def _element_view_factors(
    mesh: BeamSurfaceMesh,
    elem,
    model_nodes: dict,
    covering_zones: list[FireZone],
    n_sub: int = 4,
    n_steel_sub: int = 2,
) -> np.ndarray:
    """
    Compute FAHTS §3.3.4 geometric view factor for every quad in *mesh*.

    Implements the full double-area numerical integration:

        F_12 = (1/A1) · Σ_i Σ_j  cosθ_i · cosθ_j / (π·r²) · Ai · Aj

    Surface 1 (each steel quad) is subdivided into n_steel_sub² sub-patches;
    surface 2 (fire zone faces) is subdivided via FireZone.face_patches(n_sub).

    Parameters
    ----------
    mesh          : BeamSurfaceMesh in beam-local coordinates
    elem          : BeamElement (provides direction, local_z, n1)
    model_nodes   : {nid: Node} dict for global node positions
    covering_zones: active FireZone objects covering this element
    n_sub         : fire zone face subdivision count (default 4)
    n_steel_sub   : steel quad subdivision count per edge (default 2)

    Returns
    -------
    (n_quads,) view factor array
    """
    R = _beam_local_to_global(elem)
    origin = model_nodes[elem.n1].xyz

    # Collect sub-patches from all covering fire zones
    all_patches: list[tuple[np.ndarray, np.ndarray, float]] = []
    for zone in covering_zones:
        all_patches.extend(zone.face_patches(n_sub))

    F = np.zeros(mesh.n_quads)
    for q in range(mesh.n_quads):
        quad_local = mesh.nodes[mesh.quads[q]]           # (4, 3) beam-local
        quad_global = origin + (R @ quad_local.T).T      # (4, 3) global

        normal_local = _quad_outward_normal_local(mesh, q)
        normal_global = R @ normal_local
        norm = np.linalg.norm(normal_global)
        if norm > 1e-9:
            normal_global /= norm

        F[q] = geometric_view_factor_double_area(
            quad_global, normal_global, all_patches, n_steel_sub,
        )

    return F


def _build_beam_surface_mesh(sec, length: float, config: AnalysisConfig) -> BeamSurfaceMesh:
    """Build the FAHTS surface mesh for a beam element based on section type."""
    if isinstance(sec, BoxSection):
        return BoxSurfaceMesher(
            section=sec, length=length,
            n_top=config.n_top, n_side=config.n_side, n_length=config.n_length,
        ).build()
    if isinstance(sec, ISection):
        return IProfileSurfaceMesher(
            section=sec, length=length,
            n_top=config.n_top_i, n_side=config.n_side_i,
            n_bottom=config.n_bottom_i, n_length=config.n_length_i,
        ).build()
    if isinstance(sec, PipeSection):
        return PipeSurfaceMesher(
            section=sec, length=length,
            c_circ=config.c_circ, n_length=config.n_length_p,
        ).build()
    raise TypeError(f"Unsupported section type: {type(sec).__name__}")


def _compute_M_extra(sec, mesh, elem_length: float) -> np.ndarray | None:
    """
    Heat accumulation element capacitance for hollow BOX/PIPE meshes.

    Legacy cross-section meshes have explicit inner nodes, so the enclosed-fluid
    capacity is added there.  Active surface meshes have no inner node set; for
    those, the same total capacity is distributed over existing thermal DOFs by
    tributary surface area so the global energy balance includes §3.3.5 inertia.
    """
    if isinstance(sec, BoxSection):
        A_inner = sec.inner_height * sec.inner_width
    elif isinstance(sec, PipeSection):
        A_inner = math.pi * sec.inner_radius ** 2
    else:
        return None

    if A_inner <= 0.0:
        return None

    total_acc = A_inner * elem_length * 1200.0
    inner_ids = getattr(mesh, "inner_node_indices", None)
    if inner_ids:
        m_acc   = total_acc / len(inner_ids)
        M_extra = np.zeros(mesh.n_nodes)
        for idx in inner_ids:
            M_extra[idx] = m_acc
        return M_extra

    if isinstance(mesh, BeamSurfaceMesh):
        M_extra = np.zeros(mesh.n_nodes)
        M_extra[:] = total_acc * mesh.node_area_weights
        return M_extra

    return None


# ── Global DOF map — co-located node merging ──────────────────────────────────

def _compute_global_node_positions(
    eids: list[int],
    meshes: dict,
    model: FEMModel,
    is_surface: dict[int, bool],
) -> dict[int, np.ndarray | None]:
    """
    Return world-space (global) 3-D coordinates for every mesh node.

    Beam elements: transform beam-local coords via rotation + n1 origin.
    QUADSHEL     : PlateSurfaceMesher nodes are already in global coords.
    TRISHELL     : 1-D through-thickness mesh — no global spatial meaning → None.
    """
    positions: dict[int, np.ndarray | None] = {}
    for eid in eids:
        if eid in model.elements:
            elem   = model.elements[eid]
            R      = _beam_local_to_global(elem)
            origin = model.nodes[elem.n1].xyz.copy()
            if elem.ecc1 is not None:
                origin += np.asarray(elem.ecc1, dtype=float)
            positions[eid] = origin + (R @ meshes[eid].nodes.T).T
        elif is_surface.get(eid, False):
            positions[eid] = meshes[eid].nodes.copy()  # already global for QUADSHEL
        else:
            positions[eid] = None  # TRISHELL: 1-D mesh, skip spatial merging
    return positions


def _build_global_dof_map(
    eids: list[int],
    meshes: dict,
    global_positions: dict[int, np.ndarray | None],
    tol: float = 1e-3,
    model: FEMModel | None = None,
    struct_tol: float = 0.025,
) -> tuple[dict[int, np.ndarray], int]:
    """
    Merge co-located surface-mesh nodes across elements into shared global DOFs.

    Two merge passes are performed:

    Pass 1 — proximity (1 mm default): merges nodes that are exactly co-located
    in global space.  Handles end-to-end beam connections where the last
    axial slice of one element sits exactly on the first slice of the next.

    Pass 2 — structural-node-aware (25 mm default): for every structural node
    in the FEM model, collects ALL end-face mesh nodes from connected beam
    elements and merges pairs from *different* elements that are within
    *struct_tol* of each other.  This catches perpendicular-crossing junctions
    (e.g. two I-beams at right angles) where the web-offset means the closest
    nodes are ~tw/√2 ≈ 4–14 mm apart — outside Pass-1 tolerance but correctly
    identified because they share a structural node.

    Elements without global positions (TRISHELL) receive independent DOFs
    appended after the spatial ones.

    Returns
    -------
    gdof_map : {eid: (n_nodes,) int array}  local node → global DOF index
    n_global : total number of unique global DOFs
    """
    from scipy.spatial import cKDTree

    all_coords: list[np.ndarray] = []
    all_labels: list[tuple[int, int]] = []  # (eid, local_node_idx)

    for eid in eids:
        pos = global_positions.get(eid)
        if pos is None:
            continue
        for i in range(len(pos)):
            all_coords.append(pos[i])
            all_labels.append((eid, i))

    n_spatial = len(all_labels)
    parent = list(range(n_spatial))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # ── Pass 1: exact proximity (end-to-end connections) ─────────────────────
    if n_spatial > 0:
        tree = cKDTree(np.array(all_coords))
        for i, j in tree.query_pairs(tol):
            _union(i, j)

    # ── Pass 2: structural-node-aware merge (perpendicular junctions) ─────────
    # For each structural node, merge the closest end-face node pairs from
    # different elements within struct_tol.  Only beam elements participate;
    # the model argument enables this pass.
    if model is not None and n_spatial > 0:
        # Reverse lookup: (eid, local_idx) → flat index in all_labels
        label_to_flat: dict[tuple[int, int], int] = {
            lbl: i for i, lbl in enumerate(all_labels)
        }
        coords_arr = np.array(all_coords)

        for nid in model.nodes:
            # Collect end-face flat-indices from all analysis beam elements at nid
            end_flat: list[int] = []
            end_eids: list[int] = []

            for eid in eids:
                if eid not in model.elements:
                    continue
                elem = model.elements[eid]
                if elem.n1 != nid and elem.n2 != nid:
                    continue
                pos = global_positions.get(eid)
                if pos is None:
                    continue

                x_target = 0.0 if elem.n1 == nid else elem.length
                x_local  = meshes[eid].nodes[:, 0]
                for local_idx in np.where(np.abs(x_local - x_target) < 1e-9)[0]:
                    key = (eid, int(local_idx))
                    if key in label_to_flat:
                        end_flat.append(label_to_flat[key])
                        end_eids.append(eid)

            if len(end_flat) < 2:
                continue

            # Local proximity search among these end-face nodes
            local_pts = coords_arr[end_flat]
            local_tree = cKDTree(local_pts)
            for li, lj in local_tree.query_pairs(struct_tol):
                if end_eids[li] != end_eids[lj]:   # only across elements
                    _union(end_flat[li], end_flat[lj])

    # ── Assign global DOF indices ─────────────────────────────────────────────
    root_to_dof: dict[int, int] = {}
    dof_idx = 0
    label_to_dof: dict[tuple[int, int], int] = {}
    for i, label in enumerate(all_labels):
        root = _find(i)
        if root not in root_to_dof:
            root_to_dof[root] = dof_idx
            dof_idx += 1
        label_to_dof[label] = root_to_dof[root]

    gdof_map: dict[int, np.ndarray] = {}
    for eid in eids:
        pos     = global_positions.get(eid)
        n_nodes = meshes[eid].n_nodes
        if pos is None:
            gdof_map[eid] = np.arange(dof_idx, dof_idx + n_nodes, dtype=np.intp)
            dof_idx += n_nodes
        else:
            gdof_map[eid] = np.array(
                [label_to_dof[(eid, i)] for i in range(n_nodes)], dtype=np.intp
            )

    return gdof_map, dof_idx


# ── USFOS-style console log helpers ──────────────────────────────────────────

def _log_header(
    log_cb: Callable[[str], None],
    model: FEMModel,
    config: AnalysisConfig,
    n_beams: int,
    n_shells: int,
    total: int,
    active_zones: list,
    active_balls: list,
    active_csrcs: list | None = None,
    active_lsrcs: list | None = None,
) -> None:
    n_sources = (
        len(active_zones) + len(active_balls)
        + len(active_csrcs or []) + len(active_lsrcs or [])
    )
    n_steps   = max(1, round(config.t_end / config.dt))
    src_name  = model.source_file.name if model.source_file else "unknown"
    stamp     = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    lines: list[str] = [
        "",
        "                              - F A H T S -                  ",
        "",
        "                          Fire And Heat Transfer             ",
        "",
        "                      Simulations of Frame Structures        ",
        "",
        "",
        f"                    - Analysis initiated at -",
        f"                    Date : {stamp}",
        "",
        "",
        "     ======  A N A L Y S I S   P A R A M E T E R S  ======",
        "",
        f"               Model file                     : {src_name}",
        f"               Number of beam elements        = {n_beams:>6d}",
        f"               Number of shell elements       = {n_shells:>6d}",
        f"               Total elements to solve        = {total:>6d}",
        f"               Number of active fire sources  = {n_sources:>6d}",
        "",
        f"               Total simulation time          : {config.t_end:>10.2f}  s"
        f"  ( {config.t_end / 60.0:.2f} min )",
        f"               Time step  (dt)                : {config.dt:>10.4f}  s",
        f"               Output interval                : {config.output_dt:>10.2f}  s",
        f"               Number of time steps           : {n_steps:>6d}",
        f"               Initial temperature            : {20.0:>10.2f}  deg C",
        "",
        "",
        "         Step      Time        Max         Min ",
        "          no.      (min)     (degr C)    (degr C)",
        "",
    ]
    for line in lines:
        log_cb(line)


def _log_step_row(
    log_cb: Callable[[str], None],
    step_no: int,
    t_min: float,
    T_max: float,
    T_min: float,
) -> None:
    log_cb(f"         {step_no:>4d}       {t_min:>7.4f}     {T_max:>8.1f}    {T_min:>8.1f}")


def _log_element_section_header(log_cb: Callable[[str], None]) -> None:
    lines: list[str] = [
        "",
        "",
        "     ======  E L E M E N T   R E S U L T S  ======",
        "",
        "      Elem no    Section       T_peak",
        "      " + "-" * 38,
    ]
    for line in lines:
        log_cb(line)


def _log_element_row(
    log_cb: Callable[[str], None],
    eid: int,
    sec_type: str,
    t_peak: float,
) -> None:
    log_cb(f"       {eid:>6d}    {sec_type:<12s}  {t_peak:>8.1f} deg C")


def _log_summary(log_cb: Callable[[str], None], merged: TemperatureField) -> None:
    critical   = merged.critical_elements(T_crit=600.0)
    t_peak_all = (
        max(merged.peak_centroid_temperature(e) for e in merged.element_ids)
        if merged.element_ids else 0.0
    )

    lines: list[str] = [
        "",
        "",
        "     ======  R E S U L T S   S U M M A R Y  ======",
        "",
        f"               Elements solved                = {merged.n_elements:>6d}",
        f"               Critical (>= 600 deg C)        = {len(critical):>6d}",
        f"               Peak temperature               : {t_peak_all:>8.1f}  deg C",
        "",
        "     " + "=" * 62,
        "              A N A L Y S I S   C O M P L E T E",
        "     " + "=" * 62,
        "",
    ]
    for line in lines:
        log_cb(line)
