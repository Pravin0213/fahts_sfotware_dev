"""
Batched assembly of many SolidTransientSolver members into global K, M, Q.

Assembling member-by-member costs ~30 small numpy calls per member per Picard
iteration; with hundreds of members that Python overhead dominates the run time.
``SolidBatch`` stacks the precomputed element / face integrals of every eligible
member (hex conductivity & capacity bases, outer/inner face masses, Gauss data) into
global arrays addressed by GLOBAL DOFs once, so one assembly is a handful of large
vectorised operations.  Member-level boundary data (fire temperature, prescribed flux,
emissivities, view factors, directional flux, contents Robin BC) are broadcast to faces.

All geometry is folded into three precomputed sparse operators, because the assembled
values are LINEAR in a small per-element parameter vector:
    K_data = S_K · [k(T̄_hex) | h·exp per face | h_r per face Gauss point | h_in per face]
    M_diag = S_M · ρc(T̄_hex) (+ M_extra)
    Q      = S_Q · [linear face load | q·detJ per face Gauss point | h_in·T_in]
so each assembly is three sparse mat-vecs plus the per-element physics.

Physics is identical to ``SolidTransientSolver.assemble_raw`` (verified by
tests/test_batch_assembly.py to round-off):
    K = Σ_hex k(T̄_hex)·K_base + Σ_outer (h·exp·F_mass + radiation tangent)
        + Σ_inner h_in·F_mass
    M = Σ_hex ρ c(T̄_hex)·C_base (lumped) + M_extra
    Q = outer: h·T_fire + q_presc + q_dir + σ(a·T_fire⁴ − e·T⁴) + h_r·T*   (Gauss-integrated)
        inner: h_in·T_in

Eligible members: SolidTransientSolver with lumped mass and no insulation layer.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fahts.core.heat.solver.solid_solver import (
    _K0,
    _SIGMA,
    SolidTransientSolver,
    material_properties,
)


def is_batchable(solver) -> bool:
    return (isinstance(solver, SolidTransientSolver)
            and solver._insulation is None and solver._mass_matrix == "lumped")


class SolidBatch:
    """Global-DOF assembly of a set of batchable SolidTransientSolver members."""

    def __init__(self, eids: list[int], solvers: dict, gdof_map: dict[int, np.ndarray],
                 n_global: int) -> None:
        self.eids = list(eids)
        self._sv = [solvers[e] for e in self.eids]
        self._n = n_global
        gd = [np.asarray(gdof_map[e], dtype=np.intp) for e in self.eids]

        # ── hexes ────────────────────────────────────────────────────────────
        self.hexes = np.concatenate([g[s._hexes] for g, s in zip(gd, self._sv)])     # (H, 8)
        self.K_base = np.concatenate([s._K_base for s in self._sv])                # (H, 8, 8)
        self.C_base = np.concatenate([s._C_base for s in self._sv])                # (H, 8)
        n_h = np.array([len(s._hexes) for s in self._sv])
        hex_mem = np.repeat(np.arange(len(self._sv)), n_h)
        # material groups (by object identity) → hex index arrays
        mats: dict[int, tuple[object, list[int]]] = {}
        for m, s in enumerate(self._sv):
            mats.setdefault(id(s._mat), (s._mat, []))[1].append(m)
        self._mat_groups = [(mat, np.flatnonzero(np.isin(hex_mem, mem)))
                            for mat, mem in mats.values()]
        self.rho_h = np.array([float(s._mat.rho) for s in self._sv])[hex_mem]

        # ── outer faces ──────────────────────────────────────────────────────
        self.fo = np.concatenate([g[s._fo] for g, s in zip(gd, self._sv)])          # (F, 4)
        self.Fo_mass = np.concatenate([s._Fo_mass for s in self._sv])
        self.Fo_row = np.concatenate([s._Fo_row for s in self._sv])
        self.Fo_detJ = np.concatenate([s._Fo_detJ for s in self._sv])
        self.N_gp = self._sv[0]._N_gp if self._sv else np.eye(4)
        if any(not np.array_equal(sv._N_gp, self.N_gp) for sv in self._sv):
            raise ValueError("SolidBatch members must share one face quadrature rule "
                             "(mixed conduction schemes)")
        n_f = np.array([len(s._fo) for s in self._sv])
        self.face_mem = np.repeat(np.arange(len(self._sv)), n_f)
        self.exp = np.concatenate([s._exp for s in self._sv]) if self._sv else np.zeros(0)
        self.F_arr = np.concatenate([
            s._F_arr if s._use_vf else np.zeros(len(s._fo)) for s in self._sv
        ]) if self._sv else np.zeros(0)
        self._n_f = n_f

        # ── inner faces ──────────────────────────────────────────────────────
        self.fi = np.concatenate([g[s._fi] for g, s in zip(gd, self._sv)]).reshape(-1, 4)
        self.Fi_mass = np.concatenate([s._Fi_mass for s in self._sv]).reshape(-1, 4, 4)
        self.Fi_row = np.concatenate([s._Fi_row for s in self._sv]).reshape(-1, 4)
        n_i = np.array([len(s._fi) for s in self._sv])
        self.inner_mem = np.repeat(np.arange(len(self._sv)), n_i)

        # ── static member scalars ────────────────────────────────────────────
        self.h_conv = np.array([s._h_conv for s in self._sv])
        self.use_vf = np.array([s._use_vf for s in self._sv], dtype=bool)
        self.eps = np.array([s._eps for s in self._sv])
        self.eps_steel = np.array([float(s._eps_steel) for s in self._sv])
        self.eps_fire = np.array([float(s._eps_fire) for s in self._sv])
        self.eps_rerad = np.array([s._eps_rerad for s in self._sv])
        self.T0 = np.array([s._T0 for s in self._sv])
        self.has_qpf = np.array([s._q_per_face is not None for s in self._sv], dtype=bool)
        self.M_extra = np.zeros(n_global)
        for g, s in zip(gd, self._sv):
            if s._M_extra is not None:
                np.add.at(self.M_extra, g, s._M_extra)
        self._t_cache: float | None = None
        self._t_data: dict = {}

    # ── COO structure (global DOFs) in the order of assemble()'s K values ──────
    def k_coo(self) -> tuple[np.ndarray, np.ndarray]:
        def pairs(conn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            m = conn.shape[1]
            return np.repeat(conn, m, axis=1).ravel(), np.tile(conn, (1, m)).ravel()
        blocks = [pairs(self.hexes), pairs(self.fo), pairs(self.fi)]
        return (np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]))

    # ── per-time member data (shared by all Picard iterations of a step) ──────
    def _time_data(self, t: float) -> dict:
        if self._t_cache == t:
            return self._t_data
        sv = self._sv
        T_fire = np.array([float(s._fire_temp(t)) for s in sv])
        q_presc = np.array([float(s._q_fn(t)) for s in sv])
        q_dir = np.concatenate([
            (np.asarray(s._q_face_fn(t), float) if s._q_face_fn is not None else s._q_per_face)
            if s._q_per_face is not None else np.zeros(len(s._fo)) for s in sv
        ]) if sv else np.zeros(0)
        h_in = np.array([s._inner_bc.h_at(t) if s._inner_bc is not None else 0.0 for s in sv])
        T_in = np.array([s._inner_bc.T_fluid_at(t) if s._inner_bc is not None else 0.0
                         for s in sv])
        need_rad = self.use_vf | (self.eps != 0.0)
        need_rerad = (~need_rad) & (self.eps_rerad > 0.0) & ((q_presc != 0.0) | self.has_qpf)
        fm = self.face_mem
        T_fK4 = (T_fire + _K0) ** 4
        # radiation coefficients per face: q = σ(a·T_fire⁴ − e·T⁴)
        a_f = np.where(self.use_vf[fm],
                       self.exp * self.eps_steel[fm] * self.F_arr * self.eps_fire[fm],
                       np.where(self.eps[fm] != 0.0, self.exp * self.eps[fm], 0.0)) * T_fK4[fm]
        e_f = np.where(self.use_vf[fm], self.exp * self.eps_steel[fm],
                       np.where(self.eps[fm] != 0.0, self.exp * self.eps[fm], 0.0))
        e_f = e_f + np.where(need_rerad[fm], self.eps_rerad[fm], 0.0)
        # re-radiation to the ambient (initial) temperature, not to 0 K
        a_f = a_f + np.where(need_rerad[fm], self.eps_rerad[fm], 0.0) * (self.T0[fm] + _K0) ** 4
        # linear face load (convection + prescribed + directional)
        lin = (self.h_conv[fm] * T_fire[fm] + q_presc[fm]) * self.exp + q_dir
        self._t_data = {"a": a_f, "e": e_f, "lin": lin, "h_in": h_in[self.inner_mem],
                        "T_in": T_in[self.inner_mem],
                        "h_face": self.h_conv[fm] * self.exp}
        self._t_cache = t
        return self._t_data

    def bind(self, inv: np.ndarray, nnz: int) -> None:
        """Build the sparse operators once the global K pattern is known.

        ``inv``: global-pattern position of every k_coo() entry (in k_coo order).
        """
        H, F, Fi = len(self.hexes), len(self.fo), len(self.fi)
        n = self._n
        self._H, self._F, self._Fi = H, F, Fi
        n_par = H + 5 * F + Fi
        o_h, o_f = 64 * H, 64 * H + 16 * F
        pos_h = inv[:o_h].reshape(H, 64)
        pos_f = inv[o_h:o_f].reshape(F, 16)
        pos_i = inv[o_f:].reshape(Fi, 16)
        NN = np.einsum("gi,gj->gij", self.N_gp, self.N_gp).reshape(4, 16)   # (gp, 16)
        rows = [pos_h.ravel(), pos_f.ravel()]
        cols = [np.repeat(np.arange(H), 64), H + np.repeat(np.arange(F), 16)]
        vals = [self.K_base.reshape(H, 64).ravel(), self.Fo_mass.reshape(F, 16).ravel()]
        for g in range(4):
            rows.append(pos_f.ravel())
            cols.append(H + F + g * F + np.repeat(np.arange(F), 16))
            vals.append(np.tile(NN[g], F))
        rows.append(pos_i.ravel())
        cols.append(H + 5 * F + np.repeat(np.arange(Fi), 16))
        vals.append(self.Fi_mass.reshape(Fi, 16).ravel())
        self._S_K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                          np.concatenate(cols))),
                                  shape=(nnz, n_par))
        self._S_M = sp.csr_matrix((self.C_base.ravel(),
                                   (self.hexes.ravel(), np.repeat(np.arange(H), 8))),
                                  shape=(n, H))
        q_rows = [self.fo.ravel()]
        q_cols = [np.repeat(np.arange(F), 4)]
        q_vals = [self.Fo_row.ravel()]
        for g in range(4):
            q_rows.append(self.fo.ravel())
            q_cols.append(F + g * F + np.repeat(np.arange(F), 4))
            q_vals.append(np.tile(self.N_gp[g], F))
        q_rows.append(self.fi.ravel())
        q_cols.append(5 * F + np.repeat(np.arange(Fi), 4))
        q_vals.append(self.Fi_row.ravel())
        self._S_Q = sp.csr_matrix((np.concatenate(q_vals), (np.concatenate(q_rows),
                                                            np.concatenate(q_cols))),
                                  shape=(n, 5 * F + Fi))

    def assemble(self, T: np.ndarray, t: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K data in the bound global pattern, lumped M (n,), Q (n,)).  Call bind() first."""
        d = self._time_data(t)
        T_hex = T[self.hexes].mean(axis=1)
        k_h = np.empty(len(T_hex))
        c_h = np.empty(len(T_hex))
        for mat, idx in self._mat_groups:
            k_h[idx], c_h[idx] = material_properties(mat, T_hex[idx])

        # radiation at Gauss points with Newton tangent (as in assemble_raw)
        T_gpK = (T[self.fo] + _K0) @ self.N_gp.T                        # (F, gp)
        T3 = T_gpK ** 3
        e = d["e"][:, None]
        q_gp = _SIGMA * (d["a"][:, None] - e * T3 * T_gpK) \
            + 4.0 * e * _SIGMA * T3 * (T_gpK - _K0)
        h_r = 4.0 * e * _SIGMA * T3 * self.Fo_detJ                      # (F, gp)

        p_K = np.concatenate([k_h, d["h_face"], h_r.T.ravel(), d["h_in"]])
        p_Q = np.concatenate([d["lin"], (q_gp * self.Fo_detJ).T.ravel(),
                              d["h_in"] * d["T_in"]])
        return (self._S_K @ p_K, self._S_M @ (self.rho_h * c_h) + self.M_extra,
                self._S_Q @ p_Q)
