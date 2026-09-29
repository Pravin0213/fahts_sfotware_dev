"""Tangent-plane stability analysis (Michelsen 1982).
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import ZFLOOR


class StabilityMixin:
    """Tangent-plane stability analysis (Michelsen 1982)."""

    # ------------------------------------------------------------- stability
    def stability(self, z, T, P, lnphi_z=None, extra_trials=(), tol=1e-9, skip_aqueous=False):
        """Michelsen tangent-plane stability.  Returns (stable, w, tm, Zw).
        skip_aqueous: ignore trial phases that converge to a water-rich (x_w > 0.5)
        phase - used in free-water mode, where that phase is the (pure) aqueous phase."""
        zf = np.maximum(np.asarray(z, float), ZFLOOR)
        if lnphi_z is None:
            lnphi_z, _ = self._lnphi(zf, T, P)
        d = np.log(zf) + lnphi_z
        Kw = self._wilson(T, P)
        trials = [zf * Kw, zf / Kw]
        if skip_aqueous:      # water-free hydrocarbon trials (liquid-like first)
            for t in (zf / Kw, zf * Kw):
                t = t.copy()
                t[self.iw] = ZFLOOR
                trials.append(t)
        trials += [np.maximum(np.asarray(t, float), ZFLOOR) for t in extra_trials]
        best = (0.0, None, None)
        for W0 in trials:
            tm, w, Zw = self._stab_trial(d, np.log(W0), T, P, zf, tol)
            if skip_aqueous and w is not None and w[self.iw] > 0.5:
                continue
            if w is not None and tm < best[0]:
                best = (tm, w, Zw)
                if tm < -1e-3:
                    break
        tm, w, Zw = best
        return (w is None or tm > -1e-10), w, tm, Zw

    def _stab_trial(self, d, lnW, T, P, zf, tol):
        lnz = np.log(zf)
        Zw = None
        G_prev = None
        for it in range(80):
            W = np.exp(lnW)
            SW = W.sum()
            w = W / SW
            if it >= 6:
                lnphiw, J, Zw = self._lnphi_jac(w, T, P)
            else:
                lnphiw, Zw = self._lnphi(w, T, P)
            G = lnW + lnphiw - d
            err = np.abs(G).max()
            if err < tol:
                break
            if it >= 6 and G_prev is not None and err > 0.5 * G_prev:
                lnW = d - lnphiw            # Newton not contracting: plain SS step
                G_prev = err
                continue
            G_prev = err
            if it > 2 and np.abs(np.log(w) - lnz).max() < 1e-4:
                return 0.0, None, None               # trivial solution
            if it >= 4 and math.log(SW) > 0.1:
                break                                # clearly unstable, good enough for K init
            if it >= 6:
                sW = np.sqrt(W)
                g = sW * G
                H = np.diag(1.0 + 0.5 * G) + np.outer(sW, sW) * J / SW
                try:
                    dal = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    lnW = d - lnphiw
                    continue
                al = 2.0 * sW
                aln = al + dal
                aln = np.where(aln <= 0.0, 0.1 * al, aln)
                lnW_n = 2.0 * np.log(0.5 * aln)
                # guard: fall back to SS when Newton step is wild
                if np.abs(lnW_n - lnW).max() > 5.0:
                    lnW_n = d - lnphiw
                lnW = lnW_n
            else:
                lnW = d - lnphiw
        W = np.exp(lnW)
        SW = W.sum()
        w = W / SW
        lnphiw, Zw = self._lnphi(w, T, P)
        if np.abs(np.log(w) - lnz).max() < 1e-4:
            return 0.0, None, None
        tm = 1.0 + float(W @ (lnW + lnphiw - d - 1.0))
        return tm, w, Zw
