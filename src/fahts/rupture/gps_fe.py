"""Axisymmetric generalized-plane-strain finite-element model of a hollow cylinder with
temperature-dependent E(r) and thermal strain (linear elements, 2-point Gauss).
"""

from __future__ import annotations

import math

import numpy as np

from fahts.materials.en1993_mechanical import E_of_T, NU, eps_th


class GeneralizedPlaneStrainFE:
    """Axisymmetric generalized-plane-strain FE of a hollow cylinder.
    Unknowns: nodal radial displacements u_i and the uniform axial strain eps_z.
    Units MPa, m (forces MN per m length)."""

    def __init__(self, a, b, n_el=120, nu=NU):
        s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_el + 1)))  # graded to both surfaces
        self.r = a + (b - a) * s
        self.a, self.b, self.nu, self.n = a, b, nu, n_el
        g = 1 / math.sqrt(3)
        r1, r2 = self.r[:-1], self.r[1:]
        self.h = r2 - r1
        self.rg = np.stack(
            [0.5 * (r1 + r2) - 0.5 * g * self.h, 0.5 * (r1 + r2) + 0.5 * g * self.h]
        )  # (2, n)
        nu_ = nu
        self.D1 = np.array([[1 - nu_, nu_, nu_], [nu_, 1 - nu_, nu_], [nu_, nu_, 1 - nu_]]) / (
            (1 + nu_) * (1 - 2 * nu_)
        )

    def _B(self, e, r):
        r1, h = self.r[e], self.h[e]
        N1, N2 = (self.r[e + 1] - r) / h, (r - r1) / h
        return np.array([[-1 / h, 1 / h, 0.0], [N1 / r, N2 / r, 0.0], [0.0, 0.0, 1.0]])

    def solve(self, E_fn, e_fn, p, F_z):
        """E_fn(r)->E [MPa], e_fn(r)->thermal strain. Returns function
        stress(r_eval) -> (sr, st, sz)."""
        n = self.n + 1
        K = np.zeros((n + 1, n + 1))
        f = np.zeros(n + 1)
        for e in range(self.n):
            idx = [e, e + 1, n]
            for q in range(2):
                r = self.rg[q, e]
                B = self._B(e, r)
                D = E_fn(r) * self.D1
                w = 0.5 * self.h[e] * 2 * np.pi * r
                K[np.ix_(idx, idx)] += B.T @ D @ B * w
                f[idx] += B.T @ D @ (e_fn(r) * np.ones(3)) * w
        f[0] += p * 2 * np.pi * self.a
        f[n] += F_z
        x = np.linalg.solve(K, f)
        u, ez = x[:n], x[n]

        # radial strain at the Gauss points (superconvergent), recovered to
        # arbitrary r by linear inter/extrapolation; hoop strain u/r and eps_z
        # are taken exactly.
        rg = self.rg.T.ravel()
        er_g = np.repeat((u[1:] - u[:-1]) / self.h, 2)

        def stress(r_eval):
            out = []
            for r in np.atleast_1d(r_eval):
                e = min(max(np.searchsorted(self.r, r) - 1, 0), self.n - 1)
                ui = u[e] + (u[e + 1] - u[e]) * (r - self.r[e]) / self.h[e]
                if r <= rg[0] or r >= rg[-1]:
                    j = 0 if r <= rg[0] else len(rg) - 2
                    # extrapolate using first/last two elements' Gauss values
                    j2 = [0, 2] if j == 0 else [len(rg) - 3, len(rg) - 1]
                    x0, x1 = rg[j2]
                    y0, y1 = er_g[j2]
                    er = y0 + (y1 - y0) * (r - x0) / (x1 - x0)
                else:
                    er = np.interp(r, rg, er_g)
                eps = np.array([er, ui / r, ez]) - e_fn(r)
                out.append(E_fn(r) * self.D1 @ eps)
            return np.array(out).T

        return stress

    def solve_vec(self, r_T, T_C, T_ref_C, p, F_z):
        """Convenience: temperature nodes -> stress function (EN 1993-1-2 data)."""
        e_ref = float(eps_th(T_ref_C))
        E_fn = lambda r: E_of_T(np.interp(r, r_T, T_C))
        e_fn = lambda r: eps_th(np.interp(r, r_T, T_C)) - e_ref
        return self.solve(E_fn, e_fn, p, F_z)
