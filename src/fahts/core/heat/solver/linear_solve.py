"""
Linear solves for the Crank–Nicolson system A·ΔT = B  (A = K + 2/Δt·M, SPD).

``"cg"``     Jacobi-preconditioned conjugate gradient, warm-started (the OpenFOAM
             approach: PCG).  The 2/Δt·M term makes A strongly diagonally dominant for
             transient heat conduction, so it converges in a few tens of iterations —
             O(nnz) per iteration, versus sparse-LU fill-in that grows fast in 3-D.
             Falls back to the direct solver (with a warning) if it does not converge.
``"direct"`` SuperLU (``scipy.sparse.linalg.spsolve``) — robust reference.

Prescribed (Dirichlet) DOFs are removed by symmetric elimination so A stays SPD:
    A_ff·x_f = b_f − A_fp·x_p,   x_p given.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

log = logging.getLogger(__name__)

LINEAR_SOLVERS: frozenset[str] = frozenset({"cg", "direct"})

#: Use the parallel numba PCG only above this many unknowns — below it thread start-up
#: costs more than it saves (measured 2026-09-27: scipy faster at 8k, numba 1.7× at 64k,
#: 3× at 512k unknowns on a 20-core machine).
NUMBA_MIN_N: int = 20_000

# ── optional multi-threaded PCG kernel (numba) ────────────────────────────────
# scipy's CG runs its CSR mat-vec on one core; with numba available the whole
# Jacobi-PCG loop runs compiled and the mat-vec / dot products in parallel.
try:                                              # pragma: no cover - env dependent
    import numba as _nb

    @_nb.njit(parallel=True, cache=True, fastmath=False)
    def _pcg_jacobi_nb(indptr, indices, data, b, x, inv_d, tol2, maxiter):
        n = b.shape[0]
        r = np.empty(n)
        for i in _nb.prange(n):
            s = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                s += data[k] * x[indices[k]]
            r[i] = b[i] - s
        z = r * inv_d
        p = z.copy()
        rz = 0.0
        rr = 0.0
        for i in _nb.prange(n):
            rz += r[i] * z[i]
            rr += r[i] * r[i]
        ap = np.empty(n)
        it = 0
        while rr > tol2 and it < maxiter:
            pap = 0.0
            for i in _nb.prange(n):
                s = 0.0
                for k in range(indptr[i], indptr[i + 1]):
                    s += data[k] * p[indices[k]]
                ap[i] = s
                pap += p[i] * s
            alpha = rz / pap
            rz_new = 0.0
            rr = 0.0
            for i in _nb.prange(n):
                x[i] += alpha * p[i]
                r[i] -= alpha * ap[i]
                z[i] = r[i] * inv_d[i]
                rz_new += r[i] * z[i]
                rr += r[i] * r[i]
            beta = rz_new / rz
            rz = rz_new
            for i in _nb.prange(n):
                p[i] = z[i] + beta * p[i]
            it += 1
        return x, it, rr

    HAVE_NUMBA = True
except ImportError:                               # pragma: no cover
    HAVE_NUMBA = False


class SPDSolver:
    """Configured solver for SPD systems; counts direct-solver fallbacks."""

    def __init__(self, method: str = "cg", rtol: float = 1e-10,
                 maxiter: int | None = None) -> None:
        if method not in LINEAR_SOLVERS:
            raise ValueError(f"linear_solver must be one of {sorted(LINEAR_SOLVERS)}, "
                             f"got {method!r}")
        self.method = method
        self.rtol = float(rtol)
        self.maxiter = maxiter
        self.fallbacks = 0
        self.use_numba = True        # set False to force scipy's CG (reference path)
        self.iterations = 0          # total CG iterations (diagnostics)

    def solve(self, A: sp.csr_matrix, b: np.ndarray, x0: np.ndarray | None = None
              ) -> np.ndarray:
        """Solve A·x = b (A SPD, CSR)."""
        if self.method == "cg":
            if not np.any(b):
                return np.zeros_like(b)
            diag = A.diagonal()
            inv_d = 1.0 / np.where(diag > 0.0, diag, 1.0)
            if HAVE_NUMBA and self.use_numba and A.shape[0] >= NUMBA_MIN_N:
                maxiter = self.maxiter or max(1000, 10 * A.shape[0])
                x = (np.zeros_like(b) if x0 is None else np.array(x0, dtype=float))
                tol2 = (self.rtol * float(np.linalg.norm(b))) ** 2
                x, it, rr = _pcg_jacobi_nb(A.indptr, A.indices, A.data, b, x, inv_d,
                                           tol2, maxiter)
                self.iterations += int(it)
                if rr <= tol2:
                    return x
                self.fallbacks += 1
                log.warning("CG did not converge in %d iterations — falling back to "
                            "direct solve.", it)
                return spla.spsolve(A.tocsc(), b)
            prec = spla.LinearOperator(A.shape, matvec=lambda v: inv_d * v, dtype=float)
            count = [0]

            def _cb(_xk: np.ndarray) -> None:
                count[0] += 1

            x, info = spla.cg(A, b, x0=x0, rtol=self.rtol, atol=0.0, M=prec,
                              maxiter=self.maxiter or max(1000, 10 * A.shape[0]),
                              callback=_cb)
            self.iterations += count[0]
            if info == 0:
                return x
            self.fallbacks += 1
            log.warning("CG did not converge (info=%d) — falling back to direct solve.", info)
        return spla.spsolve(A.tocsc(), b)

    def solve_constrained(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        free: np.ndarray,
        pinned: np.ndarray,
        x_pinned: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve with x[pinned] = x_pinned by symmetric elimination (keeps A_ff SPD)."""
        if len(pinned) == 0:
            return self.solve(A, b, x0)
        x = np.zeros_like(b)
        x[pinned] = x_pinned
        if len(free):
            A_f = A[free]
            b_f = b[free] - A_f[:, pinned] @ x_pinned
            x[free] = self.solve(A_f[:, free].tocsr(), b_f, None if x0 is None else x0[free])
        return x
