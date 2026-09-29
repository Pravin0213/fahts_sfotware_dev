"""Transport properties: Chung et al. (1988) / LBC viscosity, Chung conductivity,
Macleod-Sugden surface tension, IAPWS water via CoolProp.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.thermo.constants import R


class TransportMixin:
    """Transport properties: Chung et al. (1988) / LBC viscosity, Chung conductivity,"""

    # ------------------------------------------------------------- transport
    def _chung_mix(self, x):
        s, e, w, M = self._csig, self._ceps, self.omega, self._cM
        X = np.outer(x, x)
        sij = np.sqrt(np.outer(s, s))
        eij = np.sqrt(np.outer(e, e))
        wij = 0.5 * (w[:, None] + w[None, :])
        Mij = 2.0 * np.outer(M, M) / (M[:, None] + M[None, :])
        kij = np.sqrt(np.outer(self._kap, self._kap))
        s3ij = sij ** 3
        s3 = float((X * s3ij).sum())
        sm = s3 ** (1.0 / 3.0)
        em = float((X * eij * s3ij).sum()) / s3
        wm = float((X * wij * s3ij).sum()) / s3
        Mm = (float((X * eij * sij ** 2 * np.sqrt(Mij)).sum()) / (em * sm * sm)) ** 2
        mu4 = s3 * float((X * np.outer(self._dip ** 2, self._dip ** 2) / s3ij).sum())
        km = float((X * kij).sum())
        Vcm = (sm / 0.809) ** 3
        Tcm = 1.2593 * em
        mur = 131.3 * mu4 ** 0.25 / math.sqrt(Vcm * Tcm)
        return Tcm, Vcm, wm, Mm, mur, km

    _CH_VIS = np.array([
        [6.324, 50.412, -51.680, 1189.0], [1.210e-3, -1.154e-3, -6.257e-3, 0.03728],
        [5.283, 254.209, -168.48, 3898.0], [6.623, 38.096, -8.464, 31.42],
        [19.745, 7.630, -14.354, 31.53], [-1.900, -12.537, 4.985, -18.15],
        [24.275, 3.450, -11.291, 69.35], [0.7972, 1.117, 0.01235, -4.117],
        [-0.2382, 0.06770, -0.8163, 4.025], [0.06863, 0.3479, 0.5926, -0.727]])
    _CH_TC = np.array([
        [2.4166, 0.74824, -0.91858, 121.72], [-0.50924, -1.5094, -49.991, 69.983],
        [6.6107, 5.6207, 64.760, 27.039], [14.543, -8.9139, -5.6379, 74.344],
        [0.79274, 0.82019, -0.69369, 6.3173], [-5.8634, 12.801, 9.5893, 65.529],
        [91.089, 128.11, -54.217, 523.81]])

    def viscosity(self, x, T, rho_mol, method="chung"):
        """Dynamic viscosity [Pa s] at T and molar density [mol/m3].
        The Chung mixture rules (polarity / association terms) can give non-physical
        (negative) values for water-containing hydrocarbon gases; LBC is used then."""
        x = self._z(x)
        if method.lower() == "lbc":
            return self._visc_lbc(x, T, rho_mol)
        mu = self._visc_chung(x, T, rho_mol)
        if not (math.isfinite(mu) and mu > 0.0):
            return self._visc_lbc(x, T, rho_mol)
        return mu

    def _visc_chung(self, x, T, rho_mol):
        Tcm, Vcm, wm, Mm, mur, km = self._chung_mix(x)
        Ts = 1.2593 * T / Tcm
        Om = 1.16145 * Ts ** -0.14874 + 0.52487 * math.exp(-0.77320 * Ts) \
            + 2.16178 * math.exp(-2.43787 * Ts)
        Fc = 1.0 - 0.2756 * wm + 0.059035 * mur ** 4 + km
        E = self._CH_VIS @ np.array([1.0, wm, mur ** 4, km])
        y = rho_mol * 1e-6 * Vcm / 6.0
        G1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3
        t1 = -math.expm1(-E[3] * y) / y if y > 1e-12 else E[3]
        G2 = (E[0] * t1 + E[1] * G1 * math.exp(E[4] * y) + E[2] * G1) / (E[0] * E[3] + E[1] + E[2])
        ess = E[6] * y * y * G2 * math.exp(E[7] + E[8] / Ts + E[9] / Ts ** 2)
        es = math.sqrt(Ts) / Om * Fc * (1.0 / G2 + E[5] * y) + ess
        return es * 36.344 * math.sqrt(Mm * Tcm) / Vcm ** (2.0 / 3.0) * 1e-7

    def _visc_lbc(self, x, T, rho_mol):
        Tc, Pa, M = self.Tc, self.Pc / 101325.0, self._cM
        xi = Tc ** (1 / 6) / (np.sqrt(M) * Pa ** (2 / 3))
        Tr = T / Tc
        mu = np.where(Tr <= 1.5, 34e-5 * Tr ** 0.94, 17.78e-5 * np.abs(4.58 * Tr - 1.67) ** 0.625) / xi
        sm = np.sqrt(M)
        mu0 = float((x * mu * sm).sum() / (x * sm).sum())
        xim = float(x @ Tc) ** (1 / 6) / (math.sqrt(float(x @ M)) * float(x @ Pa) ** (2 / 3))
        rr = rho_mol * float(x @ self.Vc)
        pol = 0.1023 + 0.023364 * rr + 0.058533 * rr ** 2 - 0.040758 * rr ** 3 + 0.0093324 * rr ** 4
        return (mu0 + (pol ** 4 - 1e-4) / xim) * 1e-3   # cP -> Pa s

    def thermal_conductivity(self, x, T, rho_mol):
        """Thermal conductivity [W/m/K], Chung et al. (1988)."""
        x = self._z(x)
        Tcm, Vcm, wm, Mm, mur, km = self._chung_mix(x)
        Ts = 1.2593 * T / Tcm
        Om = 1.16145 * Ts ** -0.14874 + 0.52487 * math.exp(-0.77320 * Ts) \
            + 2.16178 * math.exp(-2.43787 * Ts)
        Fc = 1.0 - 0.2756 * wm + 0.059035 * mur ** 4 + km
        eta0 = 40.785 * Fc * math.sqrt(Mm * T) / (Vcm ** (2.0 / 3.0) * Om) * 1e-7
        cv0 = float(x @ self._ig(T)[0]) - R
        al = cv0 / R - 1.5
        be = 0.7862 - 0.7109 * wm + 1.3168 * wm * wm
        Tr = T / Tcm
        Zc = 2.0 + 10.5 * Tr * Tr
        psi = 1.0 + al * (0.215 + 0.28288 * al - 1.061 * be + 0.26665 * Zc) \
            / (0.6366 + be * Zc + 1.061 * al * be)
        Bc = self._CH_TC @ np.array([1.0, wm, mur ** 4, km])
        y = rho_mol * 1e-6 * Vcm / 6.0
        G1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3
        t1 = -math.expm1(-Bc[3] * y) / y if y > 1e-12 else Bc[3]
        G2 = (Bc[0] * t1 + Bc[1] * G1 * math.exp(Bc[4] * y) + Bc[2] * G1) / (Bc[0] * Bc[3] + Bc[1] + Bc[2])
        Mp = Mm / 1000.0
        q = 3.586e-3 * math.sqrt(Tcm / Mp) / Vcm ** (2.0 / 3.0)
        return 31.2 * eta0 * psi / Mp * (1.0 / G2 + Bc[5] * y) + q * Bc[6] * y * y * math.sqrt(Tr) * G2

    def surface_tension(self, xL, rhoL_mol, yV=None, rhoV_mol=0.0):
        """Macleod-Sugden surface tension [N/m] (rho in mol/m3)."""
        xL = self._z(xL)
        yV = np.zeros(self.nc) if yV is None else self._z(yV)
        s = float(self._par @ (xL * rhoL_mol * 1e-6 - yV * rhoV_mol * 1e-6))
        return max(s, 0.0) ** 4 * 1e-3

    def transport_rho(self, p):
        """Molar density [mol/m3] used for transport / surface tension of a phase: the
        Peneloux-translated density (whether or not volume_shift is on), because the
        Chung and parachor methods need realistic liquid densities (plain PR liquid
        densities are typically 5-10 % high for light and low for heavy components)."""
        return 1.0 / (p.v + p.x @ self._c - p.x @ self._cpen)

    @staticmethod
    def water_surface_tension(T):
        """IAPWS (1994) release on the surface tension of ordinary water [N/m]."""
        tau = max(1.0 - T / 647.096, 0.0)
        return 0.2358 * tau ** 1.256 * (1.0 - 0.625 * tau)

    def _water_transport(self, T, P):
        """Pure-water viscosity / conductivity from the IAPWS formulations (IAPWS 2008
        viscosity, IAPWS 2011 conductivity) as implemented in CoolProp; None if unavailable."""
        try:
            import CoolProp.CoolProp as CP
            return CP.PropsSI("V", "T", T, "P", P, "Water"), CP.PropsSI("L", "T", T, "P", P, "Water")
        except Exception:
            return None

    def transport(self, r, method="chung", water="iapws"):
        """Add viscosity p.mu [Pa s] and conductivity p.k [W/m/K] to all phases of a
        FlashResult and the surface tension r.sigma [N/m] (vapour-HC liquid by
        Macleod-Sugden; vapour-aqueous = IAPWS pure-water value when no HC liquid).
        Hydrocarbon phases: Chung et al. (or LBC viscosity with method='lbc').
        Aqueous (free-water, pure) phase: IAPWS via CoolProp when water='iapws'
        (the Chung method is poor for water, see REPORT), else Chung."""
        for p in r.phases:
            wt = None
            if p.name == "aqueous" and water == "iapws" and self.iw is not None                     and p.x[self.iw] > 0.999:
                wt = self._water_transport(p.T, p.P)
            if wt is not None:
                p.mu, p.k = wt
            else:
                rho = self.transport_rho(p)
                p.mu = self.viscosity(p.x, p.T, rho, method)
                p.k = self.thermal_conductivity(p.x, p.T, rho)
        V, L = r.vapour, r.liquid
        if L is not None:
            r.sigma = self.surface_tension(L.x, self.transport_rho(L), V.x if V else None,
                                           self.transport_rho(V) if V else 0.0)
        elif r.aqueous is not None:
            r.sigma = self.water_surface_tension(r.T)
        return r
