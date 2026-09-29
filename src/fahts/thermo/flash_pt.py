"""Isothermal (PT) flash: two-phase flash (SS + Newton) and the public ``flash_PT``.
"""

from __future__ import annotations

import numpy as np

from fahts.thermo.constants import ZFLOOR
from fahts.thermo.numerics import rachford_rice
from fahts.thermo.results import FlashResult


class FlashPTMixin:
    """Isothermal (PT) flash: two-phase flash (SS + Newton) and the public ``flash_PT``."""

    # ------------------------------------------------------------- two-phase flash
    def _flash2(self, z, T, P, K, maxss=12):
        """Two-phase flash: successive substitution with GDEM acceleration (Crowe &
        Nishio 1975, every 5th step), then Newton (retried every 10 SS steps if it fails)."""
        lnK = np.log(K)
        beta = None
        x = y = None
        next_newton = maxss
        d_prev = None
        for it in range(300):
            K = np.exp(lnK)
            beta = rachford_rice(z, K, beta)
            if beta <= -0.5 or beta >= 1.5 or not np.isfinite(beta):
                if it > 3:
                    return None
            bb = min(max(beta, -0.5), 1.5)
            den = 1.0 + bb * (K - 1.0)
            x = z / den
            y = K * x
            x = x / x.sum()
            y = y / y.sum()
            lnphiL, ZL = self._lnphi(x, T, P)
            lnphiV, ZV = self._lnphi(y, T, P)
            lnK_n = lnphiL - lnphiV
            d = lnK_n - lnK
            err = np.abs(d).max()
            lnK = lnK_n
            if np.abs(lnK).max() < 1e-4:
                return None
            if err < 1e-10:
                break
            if it >= next_newton and 0.0 < beta < 1.0 and err < 0.1:
                res = self._flash2_newton(z, T, P, beta, x, y)
                if res is not None:
                    return res
                next_newton = it + 10
            if d_prev is not None and it % 5 == 4:
                den_ = float(d_prev @ d)
                lam = float(d @ d) / den_ if den_ != 0.0 else 0.0
                if 0.0 < lam < 0.98:
                    lnK = lnK + d * min(lam / (1.0 - lam), 20.0)
            d_prev = d
        beta = rachford_rice(z, np.exp(lnK), beta)
        if not (0.0 < beta < 1.0):
            return None
        return self._pack2(z, T, P, beta, x, y)

    def _flash2_newton(self, z, T, P, beta, x, y):
        v = beta * y
        l = z - v
        if (l <= 0).any():
            return None
        err_prev = None
        for it in range(25):
            nV = v.sum()
            nL = l.sum()
            y = v / nV
            x = l / nL
            # keep each phase on its own cubic root during Newton (avoids root jumping
            # near the three-root region); the SS stage uses the min-Gibbs root
            lnphiV, JV, ZV = self._lnphi_jac(y, T, P, "V")
            lnphiL, JL, ZL = self._lnphi_jac(x, T, P, "L")
            g = np.log(y) + lnphiV - np.log(x) - lnphiL
            err = np.abs(g).max()
            if err < 1e-10:
                break
            if err_prev is not None and err > 10.0 * err_prev:
                return None
            err_prev = err
            H = (np.diag(1.0 / y) - 1.0 + JV) / nV + (np.diag(1.0 / x) - 1.0 + JL) / nL
            try:
                dv = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return None
            vn = v + dv
            vn = np.where(vn <= 0.0, 0.1 * v, vn)
            vn = np.where(vn >= z, v + 0.9 * (z - v), vn)
            v = vn
            l = z - v
        else:
            return None
        beta = v.sum()
        if not (0.0 < beta < 1.0) or np.abs(np.log(y / x)).max() < 1e-4:
            return None
        return self._pack2(z, T, P, beta, x, y)

    def _pack2(self, z, T, P, beta, x, y):
        pV = self._phase(y, T, P, name="vapour", beta=beta)
        pL = self._phase(x, T, P, name="liquid", beta=1.0 - beta)
        if pV.rho > pL.rho:
            pV, pL = pL, pV
            pV.name, pL.name = "vapour", "liquid"
        if self.iw is not None and pL.x[self.iw] > 0.5:
            pL.name = "aqueous"
        return FlashResult(T, P, z, [pV, pL], kind="std2", K=pV.x / pL.x, names=self.names)

    def _result1(self, z, T, P, Z=None, kind="std1"):
        zf = np.maximum(z, ZFLOOR) if (z <= 0).any() else z
        ph = self._phase(zf, T, P, Z=Z)
        return FlashResult(T, P, z, [ph], kind=kind, names=self.names)

    def _flash_std(self, z, T, P, init=None, same=False):
        zf = np.maximum(z, ZFLOOR)
        if same and init is not None and init.kind == "std1":
            root = "V" if init.phases[0].name == "vapour" else "L"
            _, Zs = self._lnphi(zf, T, P, root)
            return self._result1(z, T, P, Zs)
        if init is not None and init.kind == "std2" and init.K is not None:
            res = self._flash2(zf, T, P, init.K, maxss=1)
            if res is not None:
                return res
        lnphiz, Zz = self._lnphi(zf, T, P)
        extra = [p.x for p in init.phases] if init is not None else ()
        stable, w, tm, Zw = self.stability(zf, T, P, lnphi_z=lnphiz, extra_trials=extra)
        if stable:
            return self._result1(z, T, P, Zz)
        K = w / zf if Zw > Zz else zf / w
        res = self._flash2(zf, T, P, K)
        if res is None:   # try the other trial type once (Wilson)
            res = self._flash2(zf, T, P, self._wilson(T, P))
        if res is None:
            r = self._result1(z, T, P, Zz)
            r.info["warning"] = "stability test unstable but flash converged to trivial"
            return r
        return res


    # ------------------------------------------------------------- public flashes
    def flash_PT(self, P, T, z=None, init=None, same_phases=False):
        """Isothermal flash.  init: previous FlashResult for warm start.
        same_phases=True (internal, for derivatives): assume init's phase set, no
        stability test."""
        z = self._z(z)
        if (self.iw is not None and self.free_water and 1e-12 < z[self.iw] < 1.0 - 1e-12):
            return self._flash_fw(z, T, P, init, same_phases)
        if self.iw is not None and self.free_water and z[self.iw] >= 1.0 - 1e-12:
            # pure water: use the same aqueous (IAPWS) model as the free-water phase, so a
            # water pool is thermodynamically identical whether or not traces of other
            # components are present (PR water differs by ~1.7 kJ/mol in enthalpy)
            aq = self._aq_phase(T, P, 1.0)
            return FlashResult(T, P, z, [aq], kind="water1", names=self.names)
        return self._flash_std(z, T, P, init, same_phases)
