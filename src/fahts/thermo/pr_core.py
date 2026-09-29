"""Peng-Robinson core: component set-up, k_ij, ideal-gas properties, EOS roots,
fugacity coefficients (+ composition Jacobian) and single-phase properties.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.component_data import COMPONENTS, THETA
from fahts.thermo.constants import ALIASES, B_VC_PR, D1, D2, DIPOLE, HEAVY_HC, KAPPA, KIJ, PARACHOR, P_REF, R, SQ2, T_REF
from fahts.thermo.numerics import cubic_roots
from fahts.thermo.pseudo import characterise_pseudo
from fahts.thermo.results import Phase


class PRCore:
    """EOS core: component data, k_ij, ideal gas, cubic EOS, fugacities, phase props."""

    def __init__(self, composition, pseudo=None, kij=None, volume_shift=False,
                 free_water=True, water_model="iapws"):
        pseudo = {k.upper(): v for k, v in (pseudo or {}).items()}
        if isinstance(composition, dict):
            items = [(k.upper(), float(v)) for k, v in composition.items()]
        else:
            items = [(k.upper(), 0.0) for k in composition]
        names, zs, data = [], [], []
        for nm, x in items:
            nm = ALIASES.get(nm, nm)
            if nm in pseudo:
                ps = pseudo[nm]
                sg, Tb = (ps["sg"], ps["Tb"]) if isinstance(ps, dict) else ps
                d = characterise_pseudo(sg, Tb)
            elif nm in COMPONENTS:
                d = dict(COMPONENTS[nm])
                d["pseudo"] = False
                d["parachor"] = PARACHOR.get(nm, 0.0)
            else:
                raise ValueError(f"unknown component {nm} (give pseudo properties)")
            names.append(nm)
            zs.append(x)
            data.append(d)
        self.names = names
        self.data = data
        self.nc = n = len(names)
        z = np.array(zs)
        self.z = z / z.sum() if z.sum() > 0 else z
        g = lambda key: np.array([d[key] for d in data], dtype=float)
        self.Tc, self.Pc, self.omega, self.Mw, self.Vc = g("Tc"), g("Pc"), g("omega"), g("M"), g("Vc")
        self.Tb = g("Tb")
        self.is_pseudo = np.array([d["pseudo"] for d in data])
        w = self.omega
        self._m = np.where(w <= 0.491, 0.37464 + 1.54226 * w - 0.26992 * w * w,
                           0.379642 + 1.48503 * w - 0.164423 * w * w + 0.016666 * w ** 3)
        self._ac = 0.45724 * (R * self.Tc) ** 2 / self.Pc
        self._sac = np.sqrt(self._ac)
        self._sTc = np.sqrt(self.Tc)
        self._b = 0.07780 * R * self.Tc / self.Pc
        # k_ij
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = self._default_kij(i, j)
        if kij:
            for (a, b_), val in kij.items():
                i = names.index(ALIASES.get(a.upper(), a.upper()))
                j = names.index(ALIASES.get(b_.upper(), b_.upper()))
                K[i, j] = K[j, i] = val
        self.kij = K
        self._omk = 1.0 - K
        # volume translation
        self.volume_shift = volume_shift
        zra = 0.29056 - 0.08775 * w
        self._cpen = 0.50033 * (0.25969 - zra) * R * self.Tc / self.Pc
        self._c = self._cpen.copy() if volume_shift else np.zeros(n)
        # ideal gas data
        th = np.array(THETA)
        self._theta = th
        self._c0 = np.array([d.get("c0", 0.0) for d in data])
        self._V = np.array([d.get("v", [0.0] * len(th)) for d in data])
        self._poly = np.array([d.get("poly", np.zeros(3)) for d in data])
        self._Tl = np.array([d.get("Tl", 100.0) for d in data])
        self._Th = np.array([d.get("Th", 1200.0) for d in data])
        self._h_off = np.zeros(n)
        self._s_off = np.zeros(n)
        _, h0, s0 = self._ig_raw(T_REF)
        self._h_off, self._s_off = h0, s0
        self._igT = None
        self._cT = None
        # water
        self.iw = names.index("H2O") if "H2O" in names else None
        self.free_water = free_water
        self.water_model = water_model.lower()
        self._ASw = None
        self._wcache = (None, None, None)
        if self.iw is not None and self.water_model == "iapws":
            try:
                import CoolProp.CoolProp as _CP
                self._CP = _CP
                self._ASw = _CP.AbstractState("HEOS", "Water")
            except Exception:       # CoolProp not available -> PR water
                self.water_model = "pr"
        self._nw = np.ones(n, bool)
        if self.iw is not None:
            self._nw[self.iw] = False
            self._ew = np.zeros(n)
            self._ew[self.iw] = 1.0
        # transport data
        self._dip = np.array([DIPOLE.get(nm, 0.0) for nm in names])
        self._kap = np.array([KAPPA.get(nm, 0.0) for nm in names])
        self._par = np.array([d["parachor"] for d in data])
        Vc_cm3 = self.Vc * 1e6
        self._csig = 0.809 * Vc_cm3 ** (1.0 / 3.0)
        self._ceps = self.Tc / 1.2593
        self._cM = self.Mw * 1e3

    # ------------------------------------------------------------- data helpers
    def _group(self, i):
        nm = self.names[i]
        if self.is_pseudo[i] or nm in HEAVY_HC:
            return "C7+"
        return nm

    def _default_kij(self, i, j):
        gi, gj = self._group(i), self._group(j)
        for key in ((gi, gj), (gj, gi)):
            if key in KIJ:
                return KIJ[key]
        pair = {gi, gj}
        if "H2O" in pair:
            other = (pair - {"H2O"}) or {"H2O"}
            other = other.pop()
            return 0.5 if other != "H2O" else 0.0
        if "C1" in pair and "C7+" in pair:
            # Chueh & Prausnitz (1967) form, A = 0.18, B = 6 (Whitson & Brule 2000 Eq. 4.?)
            vi, vj = self.Vc[i] * 1e6, self.Vc[j] * 1e6
            t = 2.0 * (vi * vj) ** (1 / 6) / (vi ** (1 / 3) + vj ** (1 / 3))
            return 0.18 * (1.0 - t ** 6)
        return 0.0

    def _z(self, z):
        if z is None:
            z = self.z
        elif isinstance(z, dict):
            zz = np.zeros(self.nc)
            for k_, v_ in z.items():
                zz[self.names.index(ALIASES.get(k_.upper(), k_.upper()))] = v_
            z = zz
        z = np.asarray(z, float)
        return z / z.sum()

    def _wilson(self, T, P):
        return self.Pc / P * np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))

    # ------------------------------------------------------------- ideal gas
    def _ig_raw(self, T):
        Tc_ = max(T, 20.0)
        x = self._theta / Tc_
        em1 = np.expm1(x)
        E = x * x * (em1 + 1.0) / (em1 * em1)
        cp = R * (self._c0 + self._V @ E)
        h = R * (self._c0 * Tc_ + self._V @ (self._theta / em1))
        s = R * (self._c0 * math.log(Tc_) + self._V @ (x / em1 - np.log(-np.expm1(-x))))
        if self.is_pseudo.any():
            p0, p1, p2 = self._poly.T
            Tl, Th = self._Tl, self._Th
            Tq = np.clip(T, Tl, Th)
            cpl = p0 + p1 * Tl + p2 * Tl * Tl
            cph = p0 + p1 * Th + p2 * Th * Th
            cp = cp + p0 + p1 * Tq + p2 * Tq * Tq
            h = h + p0 * Tq + p1 * Tq ** 2 / 2 + p2 * Tq ** 3 / 3 \
                + cpl * (np.minimum(T, Tl) - Tl) + cph * (np.maximum(T, Th) - Th)
            s = s + p0 * np.log(Tq) + p1 * Tq + p2 * Tq ** 2 / 2 \
                + cpl * np.log(np.minimum(T, Tl) / Tl) + cph * np.log(np.maximum(T, Th) / Th)
        return cp, h - self._h_off, s - self._s_off

    def _ig(self, T):
        if T != self._igT:
            self._igC = self._ig_raw(T)
            self._igT = T
        return self._igC

    def ideal_gas(self, T):
        """Pure-component ideal-gas cp0, h0, s0(P0) arrays [J/mol(/K)]."""
        return self._ig(T)

    # ------------------------------------------------------------- EOS core
    def _eos_T(self, T):
        if T != self._cT:
            sT = math.sqrt(T)
            sa = self._sac * (1.0 + self._m * (1.0 - sT / self._sTc))
            dsa = -self._sac * self._m / (2.0 * sT * self._sTc)
            d2sa = self._sac * self._m / (4.0 * T * sT * self._sTc)
            omk = self._omk
            aij = omk * np.outer(sa, sa)
            t = np.outer(sa, dsa)
            daij = omk * (t + t.T)
            t2 = np.outer(sa, d2sa)
            d2aij = omk * (t2 + t2.T + 2.0 * np.outer(dsa, dsa))
            self._cA = (aij, daij, d2aij)
            self._cT = T
        return self._cA

    @staticmethod
    def _zsel(A, B, phase):
        roots = cubic_roots(A, B)
        if len(roots) == 1:
            return roots[0]
        if phase == "L":
            return roots[0]
        if phase == "V":
            return roots[-1]
        best, gbest = None, None
        for Z in (roots[0], roots[-1]):
            g = Z - 1.0 - math.log(Z - B) - A / (2 * SQ2 * B) * math.log((Z + D1 * B) / (Z + D2 * B))
            if gbest is None or g < gbest:
                best, gbest = Z, g
        return best

    def _lnphi(self, x, T, P, phase=None, Z=None):
        aij = self._eos_T(T)[0]
        ais = aij @ x
        a = x @ ais
        b = x @ self._b
        RT = R * T
        A = a * P / (RT * RT)
        B = b * P / RT
        if Z is None:
            Z = self._zsel(A, B, phase)
        L = math.log((Z + D1 * B) / (Z + D2 * B))
        bb = self._b / b
        lnphi = bb * (Z - 1.0) - math.log(Z - B) - A / (2 * SQ2 * B) * (2.0 * ais / a - bb) * L
        return lnphi, Z

    def lnphi(self, x, T, P, phase=None):
        """ln fugacity coefficients and Z (phase None = min-Gibbs root, 'L' or 'V')."""
        return self._lnphi(self._z(x), T, P, phase)

    def _lnphi_jac(self, x, T, P, phase=None, Z=None):
        """ln phi and J_ij = n (d ln phi_i / d n_j)_{T,P} (n = 1).  Michelsen & Mollerup
        (2007) Ch. 2-3, reduced residual Helmholtz energy F = -n g - D(T) f / T."""
        aij = self._eos_T(T)[0]
        bi = self._b
        ais = aij @ x
        a = x @ ais
        b = x @ bi
        RT = R * T
        A = a * P / (RT * RT)
        B = b * P / RT
        if Z is None:
            Z = self._zsel(A, B, phase)
        v = Z * RT / P
        vmb = v - b
        vd1 = v + D1 * b
        vd2 = v + D2 * b
        g = math.log(1.0 - b / v)
        gB = -1.0 / vmb
        gV = b / (v * vmb)
        gBB = -1.0 / vmb ** 2
        gBV = 1.0 / vmb ** 2
        gVV = -1.0 / vmb ** 2 + 1.0 / v ** 2
        f = math.log(vd1 / vd2) / (R * b * (D1 - D2))
        fV = -1.0 / (R * vd1 * vd2)
        fB = -(f + v * fV) / b
        fVV = (1.0 / vd1 + 1.0 / vd2) / (R * vd1 * vd2)
        fBV = -(2.0 * fV + v * fVV) / b
        fBB = -(2.0 * fB + v * fBV) / b
        D = a
        FB = -gB - D * fB / T
        FD = -f / T
        Di = 2.0 * ais
        Fi = -g + FB * bi + FD * Di
        lnphi = Fi - math.log(Z)
        FnB = -gB
        FBD = -fB / T
        FBB = -gBB - D * fBB / T
        ob = np.outer(bi, Di)
        Fij = FnB * (bi[:, None] + bi[None, :]) + FBD * (ob + ob.T) + FBB * np.outer(bi, bi) \
            + FD * 2.0 * aij
        FiV = -gV + (-gBV - D * fBV / T) * bi + (-fV / T) * Di
        FVV = -gVV - D * fVV / T
        dPdV = -RT * FVV - RT / v ** 2
        dPdn = -RT * FiV + RT / v
        J = Fij + 1.0 + np.outer(dPdn, dPdn) / (RT * dPdV)
        return lnphi, J, Z

    def _label(self, x, v_eos):
        b = x @ self._b
        if v_eos > b / B_VC_PR:
            return "vapour"
        if self.iw is not None and x[self.iw] > 0.5:
            return "aqueous"
        return "liquid"

    def _phase(self, x, T, P, phase=None, Z=None, name=None, beta=1.0):
        aij, daij, d2aij = self._eos_T(T)
        ais = aij @ x
        a = x @ ais
        da = x @ daij @ x
        d2a = x @ d2aij @ x
        b = x @ self._b
        RT = R * T
        if Z is None:
            Z = self._zsel(a * P / (RT * RT), b * P / RT, phase)
        v = Z * RT / P
        vd1 = v + D1 * b
        vd2 = v + D2 * b
        L = math.log(vd1 / vd2)
        k2 = 1.0 / (2.0 * SQ2 * b)
        ures = (T * da - a) * k2 * L
        hres = ures + P * v - RT
        sres = (R * math.log(Z * (v - b) / v) if Z > 0 else float("nan")) + da * k2 * L
        cvres = T * d2a * k2 * L
        dPdT = R / (v - b) - da / (vd1 * vd2)
        dPdv = -RT / (v - b) ** 2 + 2.0 * a * (v + b) / (vd1 * vd2) ** 2
        cp0_i, h0_i, s0_i = self._ig(T)
        xp = x[x > 0]
        s_ig = x @ s0_i - R * math.log(abs(P) / P_REF) - R * float(np.sum(xp * np.log(xp)))
        cv = x @ cp0_i - R + cvres
        cp = cv - T * dPdT ** 2 / dPdv
        h = x @ h0_i + hres
        s = s_ig + sres
        c = x @ self._c
        vt = v - c
        h -= P * c
        u = h - P * vt
        M = x @ self.Mw
        w2 = -vt * vt / M * cp / cv * dPdv
        if name is None:
            name = self._label(x, v)
        return Phase(name=name, beta=beta, x=x, T=T, P=P, Z=Z, v=vt, M=M, h=h, u=u, s=s,
                     cp=cp, cv=cv, w=math.sqrt(w2) if w2 > 0 else float("nan"),
                     dPdT_v=dPdT, dPdv_T=dPdv)

    def phase_props(self, x, T, P, phase=None):
        """Single-phase properties at (T,P) for composition x (root: None/'L'/'V')."""
        return self._phase(self._z(x), T, P, phase)

    def state_TV(self, x, T, v):
        """Single-phase properties at (T, true molar volume v)."""
        x = self._z(x)
        veos = v + x @ self._c
        aij = self._eos_T(T)[0]
        a = x @ aij @ x
        b = x @ self._b
        P = R * T / (veos - b) - a / ((veos + D1 * b) * (veos + D2 * b))
        return self._phase(x, T, P, Z=P * veos / (R * T))
