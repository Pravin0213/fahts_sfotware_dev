"""Blowdown line: real-gas Fanno flow with Borda-Carnot inlet (``FannoLine``), the
ideal-gas Fanno function and screening helpers. See ``fahts.relief.blowdown``.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.relief.friction import colebrook
from fahts.relief.vapour_state import VapourState


class FannoLine:
    """Real-gas Fanno line (+ Borda-Carnot inlet) for one vessel state."""

    def __init__(self, m, x, h0, D, L, mu, rough, n):
        self.m, self.x, self.h0, self.D, self.L = m, x, h0, D, L
        self.A = 0.25 * math.pi * D * D
        self.mu, self.rough, self.n = mu, rough, n

    def fstate(self, P, G2, Tg):
        """State on the Fanno line (h + G^2 v^2/2 = h0) at P -> (vap, phi, dT/dP)."""
        m, x, h0 = self.m, self.x, self.h0
        T = Tg
        for _ in range(10):
            s = VapourState(m, x, T, P)
            F = s.h + 0.5 * G2 * s.v * s.v - h0
            dT = -F / (s.cp + G2 * s.v * s.vT)
            dT = max(min(dT, 0.1 * T), -0.1 * T)  # damped (near-critical states)
            T += dT
            if abs(dT) < 5e-3:  # 5 mK: relative error in v < 2e-5
                break
        dTdP = -((s.v - s.T * s.vT) + G2 * s.v * s.vP) / (s.cp + G2 * s.v * s.vT)
        phi = 1.0 + G2 * (s.vT * dTdP + s.vP)
        return s, phi, dTdP

    def inlet(self, P_vc, u_vc, Gp, Tg, borda=True):
        """Line-inlet state after the sudden enlargement (Newton on P1)."""
        G2 = Gp * Gp
        if not borda:
            s, phi, dTdP = self.fstate(P_vc, G2, Tg)
            return P_vc, s, phi, dTdP
        c = P_vc + Gp * u_vc
        P1 = P_vc + Gp * u_vc * 0.5
        for _ in range(12):
            s, phi, dTdP = self.fstate(P1, G2, Tg)
            F = P1 + G2 * s.v - c
            if phi <= 0.05:  # no subsonic solution close by: step up
                P1 = 0.5 * (P1 + c)
                Tg = s.T
                continue
            dP = -F / phi
            P1 += dP
            Tg = s.T + dTdP * dP
            if abs(dP) < 1e-7 * P1:
                break
        s, phi, dTdP = self.fstate(P1, G2, Tg)
        return P1, s, phi, dTdP

    def reach(self, P1, s, phi, dTdP, Gp, Pend):
        """Length from P1 to Pend, or to the choke point -> (L, choked, P_end/choke)."""
        if phi <= 0.0:
            return 0.0, True, P1
        G2 = Gp * Gp
        f = colebrook(Gp * self.D / self.mu, self.rough / self.D)
        k = 2.0 * self.D / (f * G2)
        Ps = np.linspace(P1, Pend, self.n)
        L = 0.0
        g_prev, phi_prev, P_prev, T_prev, dTdP_prev = phi * k / s.v, phi, P1, s.T, dTdP
        for P in Ps[1:]:
            s, phi, dTdP = self.fstate(P, G2, T_prev + dTdP_prev * (P - P_prev))
            g = phi * k / s.v
            if phi <= 0.0:
                Pc = P_prev + (P - P_prev) * phi_prev / (phi_prev - phi)
                return L + 0.5 * g_prev * (P_prev - Pc), True, Pc
            L += 0.5 * (g_prev + g) * (P_prev - P)
            g_prev, phi_prev, P_prev, T_prev, dTdP_prev = g, phi, P, s.T, dTdP
        return L, False, Pend


def fanno_F(M, k):
    """Ideal-gas Fanno function f L*/D (Darcy f) from Mach M to choking (Shapiro 1953)."""
    M2 = M * M
    return (1 - M2) / (k * M2) + (k + 1) / (2 * k) * math.log((k + 1) * M2 / (2 + (k - 1) * M2))


def line_clearly_open(P0, T0, P1, Pb, md, st0, ln, margin=1.5):
    k = st0.cp / st0.cv
    ZRT0 = st0.P * st0.v  # Z R T0 per kg
    G = md / ln.A
    # inlet Mach from G = P M sqrt(k/(Z R T)), T = T0/(1+(k-1)/2 M^2)
    a, b = 1e-9, 1.0
    g = lambda M: P1 * M * math.sqrt(k / (ZRT0 / (1 + 0.5 * (k - 1) * M * M)))
    if g(1.0) <= G:
        return False
    for _ in range(50):
        c = 0.5 * (a + b)
        if g(c) < G:
            a = c
        else:
            b = c
    M1 = 0.5 * (a + b)
    # Mach at which the Fanno line reaches Pb: P/P* = (1/M) sqrt((k+1)/(2+(k-1)M^2))
    pr = lambda M: math.sqrt((k + 1) / (2 + (k - 1) * M * M)) / M
    target = pr(M1) * Pb / P1
    f = colebrook(G * ln.D / ln.mu, ln.rough / ln.D)
    if target <= 1.0:  # chokes before reaching Pb
        return fanno_F(M1, k) * ln.D / f > margin * ln.L
    a, b = M1, 1.0
    for _ in range(50):
        c = 0.5 * (a + b)
        if pr(c) > target:
            a = c
        else:
            b = c
    Mb = 0.5 * (a + b)
    Lb = (fanno_F(M1, k) - fanno_F(Mb, k)) * ln.D / f
    return Lb > margin * ln.L


def orifice_line_info(ln, noz, CdA, md, Pstar, Pb, Tg0, borda):
    """For an orifice-controlled flow: line inlet pressure P1 at which the line
    exactly reaches its end (P_exit = Pb, or choked at the exit)."""
    Gp = md / ln.A
    G2 = Gp * Gp
    lo, hi = Pb, None
    s, phi, dTdP = ln.fstate(Pstar, G2, Tg0)
    hi = ln.inlet(Pstar, noz.state(Pstar)[1], Gp, Tg0, borda)[0]
    Pe_hi = Pb
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        s, phi, dTdP = ln.fstate(mid, G2, Tg0)
        Lr, ch, Pe = ln.reach(mid, s, phi, dTdP, Gp, Pb)
        if Lr >= ln.L:
            hi, Pe_hi = mid, Pe
        else:
            lo = mid
    return dict(P1=hi, P_exit=Pe_hi, dP_line=hi - Pe_hi)
