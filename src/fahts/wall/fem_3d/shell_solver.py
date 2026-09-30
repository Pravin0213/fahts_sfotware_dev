"""Transient 3-D conduction in a vessel shell (Hex8 FEM), stepped by the vessel model.

    rho c(T) dT/dt = div(k(T) grad T)                 in the steel
    boundary heat rates per surface node (lumped), supplied each step by the caller as
    linearised functions of the node temperature:
        outer surface: Q_in(T)  = fire / ambient, background and peak areas
        inner surface: Q_out(T) = convection / boiling / radiation to the contents

Time integration: implicit (backward Euler, theta = 1) with properties k, c at the element
mean temperature of the start of the step and boundary rates linearised about it (one solve
per step, as the 1-D wall column). Linear system: SPD, Jacobi-preconditioned CG
(fahts.core.heat.solver.linear_solve.SPDSolver). Kernel: fahts.core.heat.solver.fem_3d.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fahts.core.heat.solver.fem_3d import hex8_capacity_base, hex8_conductivity_base
from fahts.core.heat.solver.linear_solve import SPDSolver
from fahts.materials import SteelTable
from fahts.wall.fem_3d.shell_mesh import VesselShellMesh


class ShellConduction3D:
    """Temperature field (K) of a ``VesselShellMesh`` and its implicit time step."""

    def __init__(self, mesh: VesselShellMesh, material: SteelTable, T0: float,
                 linear_solver: str = "cg") -> None:
        self.mesh, self.mat = mesh, material
        X = mesh.nodes[mesh.hexes]                                   # (n_e, 8, 3)
        self._K_base = hex8_conductivity_base(X)                     # (n_e, 8, 8)
        self._C_base = hex8_capacity_base(X, lumped=True)            # (n_e, 8) volumes
        conn = mesh.hexes
        rows = np.repeat(conn, 8, axis=1).ravel()
        cols = np.tile(conn, (1, 8)).ravel()
        n = mesh.n_nodes
        keys = rows.astype(np.int64) * n + cols
        uniq, self._inv = np.unique(keys, return_inverse=True)
        self._rows, self._cols = (uniq // n).astype(np.int32), (uniq % n).astype(np.int32)
        # CSR pattern whose data holds (unique-entry index + 1): +1 so no entry is a zero
        pattern = sp.csr_matrix((np.arange(1, len(uniq) + 1, dtype=float),
                                 (self._rows, self._cols)), shape=(n, n))
        pattern.sum_duplicates()
        pattern.sort_indices()
        self._csr_order = pattern.data.astype(np.int64) - 1         # csr position -> uniq
        self._indptr, self._indices = pattern.indptr, pattern.indices
        self._diag_pos = np.array([self._indptr[i] + np.searchsorted(
            self._indices[self._indptr[i]:self._indptr[i + 1]], i) for i in range(n)])
        self.node_volume = np.bincount(conn.ravel(), weights=self._C_base.ravel(), minlength=n)
        self.T = np.full(n, float(T0))
        self._solver = SPDSolver(linear_solver, rtol=1e-11)
        # sensible energy table e(T) = integral of cp dT (J/kg), as the 1-D column
        self._T_tab = np.linspace(50.0, 2500.0, 9801)
        cp = material.cp_at(self._T_tab)
        self._E_tab = np.concatenate(([0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1])
                                                        * np.diff(self._T_tab))))

    # ------------------------------------------------------------------ properties
    def element_temperature(self, T: np.ndarray | None = None) -> np.ndarray:
        return (self.T if T is None else T)[self.mesh.hexes].mean(axis=1)

    def _conductivity_matrix(self, T_e: np.ndarray) -> sp.csr_matrix:
        vals = (self.mat.k_at(T_e)[:, None, None] * self._K_base).ravel()
        data_u = np.bincount(self._inv, weights=vals, minlength=len(self._rows))
        return sp.csr_matrix((data_u[self._csr_order], self._indices, self._indptr),
                             shape=(self.mesh.n_nodes, self.mesh.n_nodes))

    def capacity(self, T_e: np.ndarray) -> np.ndarray:
        """Lumped nodal heat capacity [J/K]."""
        w = (self.mat.rho * self.mat.cp_at(T_e))[:, None] * self._C_base
        return np.bincount(self.mesh.hexes.ravel(), weights=w.ravel(),
                           minlength=self.mesh.n_nodes)

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
        T_e = self.element_temperature()
        A = self._conductivity_matrix(T_e)
        C = self.capacity(T_e) / dt
        To, Ti = T0[m.outer_nodes], T0[m.inner_nodes]
        qo, dqo = outer(To)
        qi, dqi = inner(Ti)
        diag = C.copy()
        rhs = C * T0
        # outer: +[qo + dqo (T - To)]  (dqo <= 0 for a physical fire / ambient boundary)
        np.add.at(diag, m.outer_nodes, -dqo)
        np.add.at(rhs, m.outer_nodes, qo - dqo * To)
        # inner: -[qi + dqi (T - Ti)]  (dqi >= 0)
        np.add.at(diag, m.inner_nodes, dqi)
        np.add.at(rhs, m.inner_nodes, -qi + dqi * Ti)
        A.data[self._diag_pos] += diag
        self.T = self._solver.solve(A, rhs, x0=T0)
        return (qo + dqo * (self.T[m.outer_nodes] - To),
                qi + dqi * (self.T[m.inner_nodes] - Ti))

    # ------------------------------------------------------------------ diagnostics
    def energy(self) -> float:
        """Sensible energy above 0 K [J] (lumped nodal masses)."""
        e = np.interp(self.T, self._T_tab, self._E_tab)
        return float(np.sum(self.mat.rho * self.node_volume * e))

    @property
    def volume(self) -> float:
        return float(self.node_volume.sum())
