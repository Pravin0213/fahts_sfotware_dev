"""
valves.py - relief/blowdown flow through an orifice followed by a blowdown line.

Orifice : API 520 Part I gas equations (critical / sub-critical), real Z and k,
          discharge coefficient Cd.
Line    : adiabatic compressible flow with friction (Fanno flow; Shapiro 1953,
          "The Dynamics and Thermodynamics of Compressible Fluid Flow", ch. 6),
          Darcy friction factor from the Colebrook equation (commercial steel,
          roughness 45 um). The line discharges to the back pressure Pb; the exit
          chokes (M = 1) if Pb is below the exit critical pressure.

Given the vessel stagnation state (P0, T0) the flow rate is the largest mdot for
which: orifice (P0 -> P1) and line (P1 -> exit) are consistent with the exit
pressure condition. For a line much wider than the orifice this reduces to the
orifice-only API 520 flow.
"""

from __future__ import annotations

import math

R_GAS = 8.314462618


def orifice_G(P0, T0, Pd, Z, k, M):
    """Ideal (Cd = 1) orifice mass flux [kg/m2/s], API 520 gas equations."""
    if P0 <= Pd:
        return 0.0
    R = R_GAS / M
    r = Pd / P0
    rc = (2 / (k + 1)) ** (k / (k - 1))
    if r <= rc:
        return math.sqrt(k * (2 / (k + 1)) ** ((k + 1) / (k - 1))) * P0 / math.sqrt(Z * R * T0)
    F2 = math.sqrt((k / (k - 1)) * r ** (2 / k) * (1 - r ** ((k - 1) / k)) / (1 - r))
    return F2 * math.sqrt(2 * (P0 - Pd) * P0 / (Z * R * T0))


def fanno_F(Mach, k):
    """Fanno function 4f_F L*/D = f_Darcy L*/D from Mach number to choking."""
    M2 = Mach * Mach
    return (1 - M2) / (k * M2) + (k + 1) / (2 * k) * math.log((k + 1) * M2 / (2 + (k - 1) * M2))


def colebrook(Re, rel_rough):
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    f = 0.02
    for _ in range(30):
        f = (-2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
    return f


def mdot_orifice_line(P0, T0, Pb, Z, k, M, Cd, d_orif, line=None, mu=1.2e-5, rough=45e-6):
    """Mass flow [kg/s] through an orifice (Cd, d_orif) and an optional blowdown line
    dict(d=outer diameter, t=wall thickness, L=length) discharging to Pb.
    Ideal-gas relations with the vessel compressibility Z and isentropic exponent k."""
    A_o = math.pi / 4 * d_orif**2
    md_orif_max = Cd * A_o * orifice_G(P0, T0, Pb, Z, k, M)
    if not line or line.get("L", 0) <= 0 or md_orif_max <= 0:
        return md_orif_max
    D = line["d"] - 2 * line.get("t", 0.0)
    if D <= d_orif * 1.0001:
        D = d_orif * 1.0001
    A_p = math.pi / 4 * D**2
    R = R_GAS / M

    def exit_pressure(md):
        """Exit static pressure for mass flow md, or None if md is infeasible
        (orifice cannot pass it, or the line chokes before its end)."""
        G = md / A_p
        # orifice: find line-inlet static pressure P1 with Cd A_o G_orif(P0 -> P1) = md
        lo, hi = Pb * 0.2, P0
        if Cd * A_o * orifice_G(P0, T0, lo, Z, k, M) < md:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if Cd * A_o * orifice_G(P0, T0, mid, Z, k, M) >= md:
                lo = mid
            else:
                hi = mid
        P1 = hi
        # line inlet Mach number from G = P M sqrt(k / (Z R T)), T = T0 / (1 + (k-1)/2 M^2)
        def g_of(Mach):
            T = T0 / (1 + 0.5 * (k - 1) * Mach**2)
            return P1 * Mach * math.sqrt(k / (Z * R * T))
        a, b = 1e-6, 1.0
        if g_of(b) < G:
            return None                                   # inlet would need M > 1
        for _ in range(60):
            c = 0.5 * (a + b)
            if g_of(c) < G:
                a = c
            else:
                b = c
        M1 = b
        T1 = T0 / (1 + 0.5 * (k - 1) * M1**2)
        Re = G * D / mu
        fLD = colebrook(Re, rough / D) * line["L"] / D
        F1 = fanno_F(M1, k)
        if F1 < fLD:
            return None                                   # chokes inside the line
        F2 = F1 - fLD
        # exit Mach from F(M2) = F2 (subsonic branch, F decreasing in M)
        a, b = M1, 1.0
        for _ in range(60):
            c = 0.5 * (a + b)
            if fanno_F(c, k) > F2:
                a = c
            else:
                b = c
        M2 = 0.5 * (a + b)
        P2 = P1 * (M1 / M2) * math.sqrt((2 + (k - 1) * M1**2) / (2 + (k - 1) * M2**2))
        return P2, M2

    # largest feasible md with exit pressure >= Pb (bisection on md)
    lo, hi = 0.0, md_orif_max
    best = 0.0
    for _ in range(50):
        md = 0.5 * (lo + hi)
        r = exit_pressure(md)
        if r is None or r[0] < Pb:
            hi = md
        else:
            lo, best = md, md
    return best
