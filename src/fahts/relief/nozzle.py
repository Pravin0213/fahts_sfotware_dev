"""Isentropic nozzle (orifice to vena contracta): isentropes (ideal / HDI / HEM) and the
``Nozzle`` table with its critical (choked) point. See ``fahts.relief.blowdown``.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.relief.vapour_state import VapourState


def isentrope_hdi(m, x, st0, Pmin, n=12):
    """Single-phase (metastable) PR vapour isentrope from st0 down to Pmin -> P, v."""
    P = st0.P * (Pmin / st0.P) ** np.linspace(0.0, 1.0, n)
    v = np.empty(n); v[0] = st0.v
    T = np.empty(n); T[0] = st0.T
    st = st0
    for i in range(1, n):
        Tg = st.T + st.T * st.vT / st.cp * (P[i] - P[i - 1])     # dT/dP|_s = T v_T / cp
        Tg = min(max(Tg, 0.7 * st.T), st.T)
        for _ in range(4):
            st = VapourState(m, x, Tg, P[i])
            dT = Tg * (st0.s - st.s) / st.cp
            dT = max(min(dT, 0.1 * Tg), -0.1 * Tg)
            Tg += dT
            if abs(dT) < 1e-5:
                break
        # final dT < 1e-5 K: keep last properties (error in v ~ 1e-7 relative)
        v[i], T[i] = st.v, st.T
    return P, v, T


def isentrope_hem(m, x, P0, T0, Pmin, n=12):
    """Equilibrium (HEM) isentrope by PS flashes -> P, v (per kg of mixture), T."""
    r0 = m.flash_PT(P0, T0, x)
    P = P0 * (Pmin / P0) ** np.linspace(0.0, 1.0, n)
    v = [r0.v / r0.M]; T = [T0]
    init = r0
    for Pi in P[1:]:
        r = m.flash_PS(Pi, r0.s, x, T_guess=T[-1], init=init)
        init = r
        v.append(r.v / r.M); T.append(r.T)
    return P, np.array(v), np.array(T)


def isentrope_ideal(P0, v0, k, Pmin, n=12):
    """Ideal-gas isentrope p v^k = const through (P0, v0 = Z0 R T0 / P0) (API 520)."""
    P = P0 * (Pmin / P0) ** np.linspace(0.0, 1.0, n)
    return P, v0 * (P0 / P) ** (1.0 / k), None


class Nozzle:
    """Ideal (Cd = 1) nozzle from an isentrope table P[0] = P0 > ... > Pmin.
    v(P) piecewise power law, h0 - h = int_P^P0 v dP (dh = v dP at constant s)."""

    def __init__(self, P, v, nfine=240):
        self.P0 = P[0]
        self._lnP = np.log(P[::-1]).copy()            # ascending
        self._lnv = np.log(v[::-1]).copy()
        lp, lv = self._lnP, self._lnv
        e = np.diff(lv) / np.diff(lp)                  # v ~ P^e on each segment
        self._e = e
        # cumulative int v dP from P[i] up to P0 (ascending index)
        seg = np.empty(len(e))
        Pa, Pb_ = np.exp(lp[1:]), np.exp(lp[:-1])
        va = np.exp(lv[1:])
        seg = va * Pa / (e + 1.0) * (1.0 - (Pb_ / Pa) ** (e + 1.0))
        cum = np.zeros(len(lp))
        cum[:-1] = np.cumsum(seg[::-1])[::-1]
        self._cum = cum
        # critical point on a fine grid
        Pf = np.exp(np.linspace(lp[0], lp[-1], nfine))
        Gf = self._G_vec(Pf)
        i = int(np.argmax(Gf))
        if 0 < i < nfine - 1:
            x = np.log(Pf[i - 1:i + 2]); y = Gf[i - 1:i + 2]
            c = np.polyfit(x - x[1], y, 2)
            xs = -c[1] / (2 * c[0]) if c[0] < 0 else 0.0
            xs = min(max(xs, x[0] - x[1]), x[2] - x[1])
            self.Pc = math.exp(x[1] + xs)
            self.Gc = max(float(np.polyval(c, xs)), float(Gf[i]))
            self.choked = True
        else:                                          # max at Pmin: not choked above Pmin
            self.Pc, self.Gc, self.choked = float(Pf[i]), float(Gf[i]), False

    def _vdh_vec(self, P):
        lp = np.log(P)
        j = np.clip(np.searchsorted(self._lnP, lp) - 1, 0, len(self._e) - 1)
        e = self._e[j]
        Pa = np.exp(self._lnP[j + 1]); va = np.exp(self._lnv[j + 1])
        v = va * (P / Pa) ** e
        part = va * Pa / (e + 1.0) * (1.0 - (P / Pa) ** (e + 1.0))
        return v, self._cum[j + 1] + part

    def _G_vec(self, P):
        v, dh = self._vdh_vec(P)
        return np.sqrt(2.0 * np.maximum(dh, 0.0)) / v

    def state(self, P):
        """(G, u) at vena-contracta pressure P (P >= Pc)."""
        v, dh = self._vdh_vec(np.array([P]))
        G = math.sqrt(2.0 * max(float(dh[0]), 0.0)) / float(v[0])
        return G, G * float(v[0])
