"""Transient 3-D conduction in a vessel shell (Hex8 FEM), stepped by the vessel model.

    rho c(T) dT/dt = div(k(T) grad T)                 in the steel
    boundary heat rates per surface node (lumped), supplied each step by the caller as
    linearised functions of the node temperature:
        outer surface: Q_in(T)  = fire / ambient, background and peak areas
        inner surface: Q_out(T) = convection / boiling / radiation to the contents

Time integration: implicit (backward Euler, theta = 1), k at the element mean temperature of
the start of the step, boundary rates linearised about it; heat capacity: a first pass with
cp(T_old), then ``secant_passes`` passes with the secant capacity
(e(T_new) - e(T_old)) / (T_new - T_old) per node, so the stored energy equals the heat balance
(the 1-D column's lagged cp has a first-order energy error, known issue #6). Linear system: SPD, CG with a *radial line* preconditioner:
the wall is thin, so coupling through the thickness is much stronger than around / along it;
each radial node line (contiguous: k runs fastest) is solved exactly (batched Thomas), CG
handles the weak lateral coupling. Conduction operator: two-point (edge-based) Hex8
(hex8_conductivity_twopoint_base) - consistent on this orthogonal mesh, monotone, and it keeps
the stiff coupling on the radial lines (the trilinear Galerkin operator also couples nodes
diagonally across the thin wall, which the line preconditioner cannot capture).
Kernel: fahts.core.heat.solver.fem_3d.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fahts.core.heat.solver.fem_3d import hex8_capacity_base, hex8_conductivity_twopoint_base
from fahts.materials import SteelTable
from fahts.wall.fem_3d.shell_mesh import VesselShellMesh


class ShellConduction3D:
    """Temperature field (K) of a ``VesselShellMesh`` and its implicit time step."""

    def __init__(
        self,
        mesh: VesselShellMesh,
        material: SteelTable,
        T0: float,
        rtol: float = 1e-10,
        secant_passes: int = 1,
    ) -> None:
        self.mesh, self.mat = mesh, material
        X = mesh.nodes[mesh.hexes]  # (n_e, 8, 3)
        # two-point (edge) conduction: M-matrix, no over/undershoot on thin walls, and on
        # this orthogonal mesh all through-thickness coupling lies on the radial lines
        self._K_base = hex8_conductivity_twopoint_base(X)  # (n_e, 8, 8)
        self._C_base = hex8_capacity_base(X, lumped=True)  # (n_e, 8) volumes
        conn = mesh.hexes
        rows = np.repeat(conn, 8, axis=1).ravel()
        cols = np.tile(conn, (1, 8)).ravel()
        n = mesh.n_nodes
        keys = rows.astype(np.int64) * n + cols
        uniq, self._inv = np.unique(keys, return_inverse=True)
        self._rows, self._cols = (uniq // n).astype(np.int32), (uniq % n).astype(np.int32)
        # CSR pattern whose data holds (unique-entry index + 1): +1 so no entry is a zero
        pattern = sp.csr_matrix(
            (np.arange(1, len(uniq) + 1, dtype=float), (self._rows, self._cols)), shape=(n, n)
        )
        pattern.sum_duplicates()
        pattern.sort_indices()
        self._csr_order = pattern.data.astype(np.int64) - 1  # csr position -> uniq
        self._indptr, self._indices = pattern.indptr, pattern.indices
        self._diag_pos = np.array(
            [
                self._indptr[i]
                + np.searchsorted(self._indices[self._indptr[i] : self._indptr[i + 1]], i)
                for i in range(n)
            ]
        )
        self.node_volume = np.bincount(conn.ravel(), weights=self._C_base.ravel(), minlength=n)
        self.T = np.full(n, float(T0))
        self.rtol = rtol
        self.secant_passes = secant_passes
        self.cg_iterations = 0
        # CSR positions of the radial neighbour entries (n, n+1) within each line
        nr = mesh.nr
        first = np.arange(n - 1)
        first = first[(first % nr) != nr - 1]  # (k, k+1) in one line
        self._off_rows = first
        self._off_pos = np.array(
            [
                self._indptr[i]
                + np.searchsorted(self._indices[self._indptr[i] : self._indptr[i + 1]], i + 1)
                for i in first
            ]
        )
        # sensible energy table e(T) = integral of cp dT (J/kg), as the 1-D column
        self._T_tab = np.linspace(50.0, 2500.0, 9801)
        cp = material.cp_at(self._T_tab)
        self._E_tab = np.concatenate(
            ([0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(self._T_tab)))
        )

    # ------------------------------------------------------------------ properties
    def element_temperature(self, T: np.ndarray | None = None) -> np.ndarray:
        return (self.T if T is None else T)[self.mesh.hexes].mean(axis=1)

    def _conductivity_matrix(self, T_e: np.ndarray) -> sp.csr_matrix:
        vals = (self.mat.k_at(T_e)[:, None, None] * self._K_base).ravel()
        data_u = np.bincount(self._inv, weights=vals, minlength=len(self._rows))
        return sp.csr_matrix(
            (data_u[self._csr_order], self._indices, self._indptr),
            shape=(self.mesh.n_nodes, self.mesh.n_nodes),
        )

    def capacity(self, T: np.ndarray | None = None) -> np.ndarray:
        """Lumped nodal heat capacity [J/K], cp at the NODE temperature: consistent with the
        nodal energy bookkeeping (``energy``); cp at the element mean left a time-step
        independent energy error of ~1 % across steep through-wall gradients."""
        T = self.T if T is None else T
        return self.mat.rho * self.mat.cp_at(T) * self.node_volume

    # ------------------------------------------------------------------ time step
    def step(self, dt: float, outer, inner) -> tuple[np.ndarray, np.ndarray]:
        """Advance one step.

        outer(T_outer_nodes) -> (Q_in [W], dQ_in/dT [W/K]) per outer surface node
        inner(T_inner_nodes) -> (Q_out [W], dQ_out/dT [W/K]) per inner surface node
        Returns the rates linearised to the new temperatures (Q_in_new, Q_out_new), which
        are what entered / left the wall over the step (energy bookkeeping).
        """
        m = self.mesh
        T0 = self.T
        K = self._conductivity_matrix(self.element_temperature())
        To, Ti = T0[m.outer_nodes], T0[m.inner_nodes]
        qo, dqo = outer(To)
        qi, dqi = inner(Ti)
        bnd_diag = np.zeros(m.n_nodes)
        bnd_rhs = np.zeros(m.n_nodes)
        # outer: +[qo + dqo (T - To)]  (dqo <= 0 for a physical fire / ambient boundary)
        np.add.at(bnd_diag, m.outer_nodes, -dqo)
        np.add.at(bnd_rhs, m.outer_nodes, qo - dqo * To)
        # inner: -[qi + dqi (T - Ti)]  (dqi >= 0)
        np.add.at(bnd_diag, m.inner_nodes, dqi)
        np.add.at(bnd_rhs, m.inner_nodes, -qi + dqi * Ti)
        # pass 1: cp(T_old); pass 2: secant capacity (e(T_new) - e(T_old)) / (T_new - T_old),
        # which makes the stored energy change equal the heat balance exactly
        C = self.capacity(T0) / dt
        T_new = T0
        for _ in range(1 + self.secant_passes):
            A = K.copy()
            A.data[self._diag_pos] += C + bnd_diag
            T_new = self._solve(A, C * T0 + bnd_rhs, T_new)
            C = self._secant_capacity(T0, T_new) / dt
        self.T = T_new
        return (qo + dqo * (self.T[m.outer_nodes] - To), qi + dqi * (self.T[m.inner_nodes] - Ti))

    def _secant_capacity(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        dT = T1 - T0
        e0, e1 = (np.interp(T, self._T_tab, self._E_tab) for T in (T0, T1))
        small = np.abs(dT) < 1e-9
        cp = np.where(small, self.mat.cp_at(T0), (e1 - e0) / np.where(small, 1.0, dT))
        return self.mat.rho * cp * self.node_volume

    # ------------------------------------------------------------------ linear solve
    def _line_preconditioner(self, A: sp.csr_matrix) -> spla.LinearOperator:
        """Exact solve of every radial line (tridiagonal, batched Thomas algorithm)."""
        nr, n = self.mesh.nr, self.mesh.n_nodes
        nl = n // nr
        d = A.data[self._diag_pos].reshape(nl, nr).copy()
        off = np.zeros((nl, nr - 1))
        off.reshape(-1)[:] = A.data[self._off_pos]  # symmetric: sub = super
        c = np.empty((nl, nr - 1))  # forward elimination
        denom = d.copy()
        for k in range(1, nr):
            c[:, k - 1] = off[:, k - 1] / denom[:, k - 1]
            denom[:, k] = d[:, k] - c[:, k - 1] * off[:, k - 1]

        def apply(r):
            y = np.asarray(r, float).reshape(nl, nr).copy()
            for k in range(1, nr):
                y[:, k] -= c[:, k - 1] * y[:, k - 1]
            y[:, -1] /= denom[:, -1]
            for k in range(nr - 2, -1, -1):
                y[:, k] = (y[:, k] - off[:, k] * y[:, k + 1]) / denom[:, k]
            return y.ravel()

        return spla.LinearOperator((n, n), matvec=apply, dtype=float)

    def _solve(self, A: sp.csr_matrix, b: np.ndarray, x0: np.ndarray) -> np.ndarray:
        count = [0]

        def cb(_x):
            count[0] += 1

        x, info = spla.cg(
            A,
            b,
            x0=x0,
            rtol=self.rtol,
            atol=0.0,
            M=self._line_preconditioner(A),
            maxiter=2000,
            callback=cb,
        )
        self.cg_iterations += count[0]
        if info != 0:
            return spla.spsolve(A.tocsc(), b)
        return x

    # ------------------------------------------------------------------ diagnostics
    def energy(self) -> float:
        """Sensible energy above 0 K [J] (lumped nodal masses)."""
        e = np.interp(self.T, self._T_tab, self._E_tab)
        return float(np.sum(self.mat.rho * self.node_volume * e))

    @property
    def volume(self) -> float:
        return float(self.node_volume.sum())
