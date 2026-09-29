"""Free-water (three-phase) flash: pure aqueous phase (IAPWS via CoolProp or PR) plus
hydrocarbon phases with water content from fugacity equality.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import P_REF, R, ZFLOOR
from fahts.thermo.results import FlashResult, Phase


class FreeWaterMixin:
    """Free-water (three-phase) flash: pure aqueous phase (IAPWS via CoolProp or PR) plus"""

    # ------------------------------------------------------------- free water
    def _water_iapws(self, T, P):
        """Pure water at (T,P) from IAPWS-95 (CoolProp): returns (lnphi, Phase-kwargs)."""
        if self._wcache[0] == T and self._wcache[1] == P:
            return self._wcache[2]
        AS, CP_ = self._ASw, self._CP
        TM = 273.16
        if T < TM:
            # Below the triple point free water would freeze (ice / hydrates are NOT
            # modelled): treat it as supercooled liquid with properties extrapolated
            # from 273.16 K at constant cp and density; ln f from the Gibbs-Helmholtz
            # relation with constant residual enthalpy.
            lnphim, kw = self._water_iapws(TM, P)
            hres = kw["h"] - self._ig(TM)[1][self.iw]
            lnf = lnphim + math.log(P) + hres / R * (1.0 / T - 1.0 / TM)
            kw = dict(kw)
            kw["h"] = kw["h"] + kw["cp"] * (T - TM)
            kw["s"] = kw["s"] + kw["cp"] * math.log(T / TM)
            kw["u"] = kw["h"] - P * kw["v"]
            kw["Z"] = P * kw["v"] / (R * T)
            out = (lnf - math.log(P), kw)
            self._wcache = (T, P, out)
            return out
        try:
            AS.update(CP_.PT_INPUTS, P, T)
        except ValueError:
            # (T, P) exactly on the saturation line: evaluate just on the liquid side
            T = T * (1.0 - 2e-6)
            AS.update(CP_.PT_INPUTS, P, T)
        Z = AS.compressibility_factor()
        lnphi = AS.alphar() + Z - 1.0 - math.log(Z)
        cp0_i, h0_i, s0_i = self._ig(T)
        iw = self.iw
        v = 1.0 / AS.rhomolar()
        h = h0_i[iw] + AS.hmolar_residual()
        s = s0_i[iw] - R * math.log(P / P_REF) + AS.smolar_residual() + R * math.log(Z)
        cp0w = AS.cp0molar()
        cp = AS.cpmolar() - cp0w + cp0_i[iw]
        cv = AS.cvmolar() - cp0w + cp0_i[iw]
        kw = dict(Z=Z, v=v, M=self.Mw[iw], h=h, u=h - P * v, s=s, cp=cp, cv=cv,
                  w=AS.speed_sound(), dPdT_v=float("nan"), dPdv_T=AS.first_partial_deriv(
                      CP_.iP, CP_.iDmolar, CP_.iT) * (-1.0 / v ** 2))
        out = (lnphi, kw)
        self._wcache = (T, P, out)
        return out

    def _aq_phase(self, T, P, beta, Zw0=None):
        if self.water_model == "iapws":
            kw = self._water_iapws(T, P)[1]
            return Phase(name="aqueous", beta=beta, x=self._ew.copy(), T=T, P=P, **kw)
        return self._phase(self._ew.copy(), T, P, Z=Zw0, name="aqueous", beta=beta)

    def _lnfw0(self, T, P):
        if self.water_model == "iapws":
            return self._water_iapws(T, P)[0] + math.log(P), None
        lnphi, Zw = self._lnphi(self._ew, T, P)
        return lnphi[self.iw] + math.log(P), Zw

    def _fw_single(self, z, T, P, lnfw0):
        iw = self.iw
        S = z[self._nw].sum()
        lnP = math.log(P)
        yw = min(math.exp(lnfw0 - lnP), 0.9)
        y = z.copy()
        for _ in range(100):
            y = z.copy()
            y[self._nw] *= (1.0 - yw) / S
            y[iw] = yw
            lnphi, Zy = self._lnphi(y, T, P)
            ywn = math.exp(lnfw0 - lnP - lnphi[iw])
            if ywn >= 1.0:
                return None
            if abs(ywn - yw) <= 1e-12 * yw:
                yw = ywn
                break
            yw = ywn
        y = z.copy()
        y[self._nw] *= (1.0 - yw) / S
        y[iw] = yw
        lnphi, Zy = self._lnphi(y, T, P)
        bHC = S / (1.0 - yw)
        bW = 1.0 - bHC
        if bW <= 1e-14:
            return None
        return y, bHC, bW, lnphi, Zy

    @staticmethod
    def _fw_rr(zn, Kn, xw, yw, bV, bL):
        for _ in range(60):
            den = bL + bV * Kn
            if (den <= 0).any():
                return False, bV, bL
            t = zn / den
            E1 = t.sum() + xw - 1.0
            E2 = (Kn * t).sum() + yw - 1.0
            t2 = t / den
            a11 = -t2.sum()
            a12 = -(t2 * Kn).sum()
            a22 = -(t2 * Kn * Kn).sum()
            det = a11 * a22 - a12 * a12
            if det == 0.0:
                return False, bV, bL
            dL = (-E1 * a22 + E2 * a12) / det
            dV = (-E2 * a11 + E1 * a12) / det
            s = 1.0
            for _k in range(40):
                if ((bL + s * dL) + (bV + s * dV) * Kn > 0).all():
                    break
                s *= 0.5
            bL += s * dL
            bV += s * dV
            if abs(E1) + abs(E2) < 1e-14:
                return True, bV, bL
        return abs(E1) + abs(E2) < 1e-9, bV, bL

    def _fw_two(self, z, T, P, lnfw0, K, betas=None):
        iw, nw = self.iw, self._nw
        zn = z[nw]
        Kn = np.maximum(K[nw], 1e-300)
        lnP = math.log(P)
        yw = min(math.exp(lnfw0 - lnP), 0.5)
        xw = 1e-3 * yw
        S = zn.sum()
        bV, bL = betas if betas is not None else (0.5 * S, 0.5 * S)
        x = y = None
        next_newton = 3
        for it in range(500):
            ok, bV, bL = self._fw_rr(zn, Kn, xw, yw, bV, bL)
            if not ok:
                return None
            den = bL + bV * Kn
            x = np.empty_like(z)
            y = np.empty_like(z)
            x[nw] = zn / den
            y[nw] = Kn * zn / den
            x[iw] = xw
            y[iw] = yw
            x /= x.sum()
            y /= y.sum()
            lnphiL, _ = self._lnphi(x, T, P)
            lnphiV, _ = self._lnphi(y, T, P)
            Kn_n = np.exp(lnphiL[nw] - lnphiV[nw])
            xw_n = math.exp(lnfw0 - lnP - lnphiL[iw])
            yw_n = math.exp(lnfw0 - lnP - lnphiV[iw])
            err = max(np.abs(np.log(Kn_n / Kn)).max(), abs(math.log(xw_n / xw)),
                      abs(math.log(yw_n / yw)))
            Kn, xw, yw = Kn_n, min(xw_n, 0.999), min(yw_n, 0.999)
            if np.abs(np.log(Kn)).max() < 1e-4:
                return None
            if err < 1e-10:
                break
            if it > 30 and (bV < -0.2 or bL < -0.2):
                break
            if it >= next_newton and bV > 0 and bL > 0 and err < 0.1:
                res = self._fw_newton(z, T, P, lnfw0, bV * y, bL * x)
                if res is not None:
                    return res
                next_newton = it + 15
        bW = 1.0 - bV - bL
        return dict(bV=bV, bL=bL, bW=bW, x=x, y=y)

    def _fw_newton(self, z, T, P, lnfw0, v, l):
        """Newton for vapour + HC liquid + pure water.  Unknowns: vapour moles v_i (all
        components) and water moles in the HC liquid l_w; l_i = z_i - v_i (i != w).
        Equations: ln f_i^V = ln f_i^L (i != w), ln f_w^V = ln f_w^L = ln f_w(pure)."""
        iw, nw = self.iw, self._nw
        n = self.nc
        lnP = math.log(P)
        v = v.copy()
        lw = l[iw]
        idx = np.flatnonzero(nw)
        err_prev = None
        for it in range(25):
            l = z - v
            l[iw] = lw
            nV, nL = v.sum(), l.sum()
            y, x = v / nV, l / nL
            lnphiV, JV, _ = self._lnphi_jac(y, T, P, "V")
            lnphiL, JL, _ = self._lnphi_jac(x, T, P, "L")
            lfV = np.log(y) + lnphiV
            lfL = np.log(x) + lnphiL
            g = np.empty(n + 1)
            g[:n - 1] = (lfV - lfL)[idx]
            g[n - 1] = lfV[iw] + lnP - lnfw0
            g[n] = lfL[iw] + lnP - lnfw0
            err = np.abs(g).max()
            if err < 1e-10:
                break
            if err_prev is not None and err > 10.0 * err_prev:
                return None
            err_prev = err
            FV = (np.diag(1.0 / y) - 1.0 + JV) / nV
            FL = (np.diag(1.0 / x) - 1.0 + JL) / nL
            Jm = np.zeros((n + 1, n + 1))
            FLc = FL.copy()
            FLc[:, iw] = 0.0                     # l_w does not depend on v_w
            Jm[:n - 1, :n] = (FV + FLc)[idx]
            Jm[:n - 1, n] = -FL[idx, iw]
            Jm[n - 1, :n] = FV[iw]
            Jm[n, :n] = -FLc[iw]
            Jm[n, n] = FL[iw, iw]
            try:
                du = np.linalg.solve(Jm, -g)
            except np.linalg.LinAlgError:
                return None
            vn = v + du[:n]
            lwn = lw + du[n]
            vn = np.where(vn <= 0.0, 0.1 * v, vn)
            upper = z.copy()
            vn[nw] = np.where(vn[nw] >= upper[nw], v[nw] + 0.9 * (upper[nw] - v[nw]), vn[nw])
            lwn = lwn if lwn > 0.0 else 0.1 * lw
            v, lw = vn, lwn
        else:
            return None
        l = z - v
        l[iw] = lw
        bV, bL = v.sum(), l.sum()
        return dict(bV=bV, bL=bL, bW=1.0 - bV - bL, x=l / bL, y=v / bV)

    def _flash_fw(self, z, T, P, init=None, same=False):
        iw = self.iw
        zf = np.maximum(z, ZFLOOR)
        lnfw0, Zw0 = self._lnfw0(T, P)

        def pack(hc_phases, bW, kind, K=None):
            phases = list(hc_phases)
            if bW > 0:
                pw = self._aq_phase(T, P, bW, Zw0)
                phases.append(pw)
            return FlashResult(T, P, z, phases, kind=kind, K=K, names=self.names)

        def three(res):
            pV = self._phase(res["y"], T, P, name="vapour", beta=res["bV"])
            pL = self._phase(res["x"], T, P, name="liquid", beta=res["bL"])
            if pV.rho > pL.rho:
                pV, pL = pL, pV
                pV.name, pL.name = "vapour", "liquid"
            return pack([pV, pL], res["bW"], "fw2", K=pV.x / pL.x)

        def ok3(res):
            return res is not None and res["bV"] > 0 and res["bL"] > 0 and res["bW"] > 0

        if init is not None and init.kind == "fw2" and init.K is not None:
            v0 = init.beta_V * init.vapour.x
            l0 = init.beta_L * init.liquid.x
            res = None
            if (v0[self._nw] < zf[self._nw]).all():
                res = self._fw_newton(zf, T, P, lnfw0, v0, l0)
            if not ok3(res):
                res = self._fw_two(zf, T, P, lnfw0, init.K, (init.beta_V, init.beta_L))
            if ok3(res):
                return three(res)
        single = self._fw_single(zf, T, P, lnfw0)
        if single is None:
            return self._flash_std(z, T, P, init if (init is not None and init.kind == "std2") else None)
        y, bHC, bW, lnphiy, Zy = single
        if same and init is not None and init.kind == "fw1":
            ph = self._phase(y, T, P, Z=Zy, beta=bHC)
            if ph.name == "aqueous":
                ph.name = "liquid"
            return pack([ph], bW, "fw1")
        extra = [p.x for p in init.phases if p.name != "aqueous"] if init is not None else ()
        stable, w, tm, Zw = self.stability(y, T, P, lnphi_z=lnphiy, extra_trials=extra,
                                           skip_aqueous=True)
        if stable:
            ph = self._phase(y, T, P, Z=Zy, beta=bHC)
            if ph.name == "aqueous":
                ph.name = "liquid"
            return pack([ph], bW, "fw1")
        K = w / y if Zw > Zy else y / w
        res = self._fw_two(zf, T, P, lnfw0, K)
        if ok3(res):
            return three(res)
        if res is not None and res["bW"] <= 0:
            std = self._flash_std(z, T, P)
            fw_ok = all(math.log(max(p.x[iw], 1e-300)) + self._lnphi(p.x, T, P, Z=p.Z)[0][iw]
                        + math.log(P) <= lnfw0 + 1e-8 for p in std.phases)
            if fw_ok:
                return std
        ph = self._phase(y, T, P, Z=Zy, beta=bHC)
        r = pack([ph], bW, "fw1")
        r.info["warning"] = "HC phase unstable but 3-phase free-water flash failed"
        return r
