"""
valves2.py - blowdown flow through an orifice (BDV) followed by a blowdown line,
real-gas (Peng-Robinson) formulation.  New module; valves.py is left unchanged.
Written from published physics only; nothing here was fitted to VessFire output.

Model (function ``mdot``)
-------------------------
Stagnation state: vessel gas (P0, T0, gas-phase composition y), velocity 0.

1. Orifice -> vena contracta (area Cd*A_o; VessFire calls Cd the
   "contraction/discharge coefficient").  Isentropic nozzle:
       G(P) = sqrt(2 [h0 - h(P, s0)]) / v(P, s0),     mdot = Cd A_o G(P_vc)
   choked when G reaches its maximum along the isentrope (general definition of
   critical flow).  Options ``nozzle=``:
     "api520"  ideal-gas isentrope with the vessel compressibility Z0 and
               k = cp/cv (real, PR) -> exactly the API 520 Part I (10th ed. 2020)
               gas equations (critical Eq. 9 / subcritical Eq. 19 form).
     "hdi"     real-gas isentrope of the (metastable, single-phase) vapour from the
               PR EOS, integrated directly ("homogeneous direct integration",
               API 520 Part I Annex B; ISO 9300:2005 real-gas critical flow
               function).  Frozen composition: no condensation in the ~ms
               residence time of an orifice (supersaturated vapour; Wilson-line
               argument, e.g. Bakhtar et al. 2005 Proc. IMechE C 219, 1315).
     "hem"     homogeneous-equilibrium isentrope from PS flashes (API 520 Annex C,
               Leung 1986 AIChE J. 32, 1743) - slow, for checks.
   Along the isentrope dh = v dP; v(P) is interpolated as piecewise power law
   (exact for a polytropic segment), so h0 - h is integrated analytically.

2. Vena contracta -> line inlet: sudden (Borda-Carnot) enlargement from Cd*A_o to
   the line area A_p, momentum balance with the vena-contracta pressure acting on
   the whole section (Borda-Carnot control volume, e.g. Shapiro 1953, "Dynamics and
   Thermodynamics of Compressible Fluid Flow", Vol. I; Benedict, Carlucci & Swetz 1966,
   J. Eng. Power 88, 73; incompressible limit K = (1 - Cd A_o/A_p)^2, Idelchik
   1986 diagram 4-1 / Crane TP-410):
       P1 + G_p^2 v1 = P_vc + G_p u_vc,     h1 + (G_p v1)^2 / 2 = h0
   (``expansion="none"`` sets P1 = P_vc: the classic "orifice discharges into the
   line pressure" approximation, valid only for A_p >> Cd A_o.)

3. Line (dict(d, t, L)) = adiabatic, constant-area, frictional real-gas flow
   (generalised Fanno flow; Shapiro 1953 ch. 6):
       G = const,  h + G^2 v^2/2 = h0,  dP + G^2 dv = - f G^2 v/(2D) dx,
   integrated in pressure with PR vapour properties; along the Fanno line
   dT/dP = -[(v - T v_T) + G^2 v v_P] / (cp + G^2 v v_T).  Choking where
   phi = 1 + G^2 (dv/dP)_Fanno = 0 (local Mach 1).  Darcy f from Colebrook (1939)
   J. ICE 11, 133 (Haaland 1983 start), roughness 45 um (commercial steel, Moody
   1944), viscosity Chung et al. (1988) at the vessel state.
   The line ends at Pb (subsonic exit, P_exit = Pb) or chokes at its end
   (P_exit >= Pb).  The flow is the largest mdot consistent with 1-3.
   ``line_id="d-2t"`` (default): the #Blowdown_line diameter is taken as the OUTER
   diameter, D = d - 2t;  ``line_id="d"``: it is the inner diameter (analogy with
   the manual's #Pipe keyword).  The default is the interpretation under which
   this physics agrees with VessFire (study 14: RMS 7-9 % vs 24-37 % for the LPG
   cases); it is an input-convention choice, not a fitted parameter - see
   studies/14_blowdown_line/REPORT.txt.

Recommended defaults: nozzle="api520", expansion="borda", line_id="d-2t".
Speed (time stepping, warm-started): ~0.5 ms when an ideal-gas screen shows that
the orifice controls with margin, ~1-4 ms otherwise (hdi adds ~3 ms); first call
of a line-limited state ~10-25 ms.  Dense vessel states (Z < 0.5) and any
integration failure fall back to valves.py (ideal-gas API 520 + Fanno).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from thermo_pr import PRMixture
except ImportError:                               # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from thermo_pr import PRMixture

R_GAS = 8.314462618
ROUGH = 45e-6
Z_DENSE = 0.5          # below this vessel Z the fallback (ideal-gas) model is used

_MODELS: dict = {}
_WARM: dict = {}


# ----------------------------------------------------------------------------- helpers
def colebrook(Re, rel_rough):
    """Darcy friction factor, Colebrook (1939); laminar 64/Re below Re = 2300."""
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    f = (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2   # Haaland (1983)
    for _ in range(6):
        f = (-2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
    return f


def get_model(names, pseudo=None):
    """PRMixture for a component list (cached)."""
    key = (tuple(n.upper() for n in names),
           tuple(sorted((k, tuple(sorted(v.items()))) for k, v in (pseudo or {}).items())))
    m = _MODELS.get(key)
    if m is None:
        m = PRMixture({n: 1.0 for n in names}, pseudo=pseudo)
        _MODELS[key] = m
    return m


class _Vap:
    """Per-mass vapour properties at (T, P) from the PR vapour root."""
    __slots__ = ("T", "P", "v", "h", "s", "cp", "cv", "vT", "vP", "w", "Z", "M", "rho_mol")

    def __init__(self, m, x, T, P):
        p = m.phase_props(x, T, P, "V")
        M = p.M
        self.T, self.P, self.M, self.Z = T, P, M, p.Z
        self.v = p.v / M
        self.h = p.h / M
        self.s = p.s / M
        self.cp = p.cp / M
        self.cv = p.cv / M
        self.vT = -p.dPdT_v / p.dPdv_T / M          # (dv/dT)_P  per kg
        self.vP = 1.0 / p.dPdv_T / M                # (dv/dP)_T  per kg
        self.w = p.w
        self.rho_mol = 1.0 / (p.v + x @ m._c)


# ----------------------------------------------------------------------------- isentropes
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
            st = _Vap(m, x, Tg, P[i])
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


# ----------------------------------------------------------------------------- line
class _Line:
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
            s = _Vap(m, x, T, P)
            F = s.h + 0.5 * G2 * s.v * s.v - h0
            dT = -F / (s.cp + G2 * s.v * s.vT)
            dT = max(min(dT, 0.1 * T), -0.1 * T)    # damped (near-critical states)
            T += dT
            if abs(dT) < 5e-3:          # 5 mK: relative error in v < 2e-5
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
            if phi <= 0.05:              # no subsonic solution close by: step up
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


# ----------------------------------------------------------------------------- main
def _parse(y, names):
    if isinstance(y, dict):
        names = list(y.keys())
        x = np.array([y[k] for k in names], float)
    else:
        x = np.asarray(y, float)
    x = np.maximum(x, 0.0)
    return list(names), x / x.sum()


def mdot(P0, T0, y, Pb, Cd, d_orifice, line=None, model=None, names=None, pseudo=None,
         nozzle="api520", expansion="borda", line_id="d-2t", rough=ROUGH, n_iso=12,
         n_line=8, info=False):
    """Blowdown mass flow [kg/s] (see module docstring).

    Robustness: if the real-gas line/nozzle integration fails (e.g. a near-critical
    dense vessel state), the ideal-gas API 520 + Fanno model of valves.py is used
    with the PR Z and cp/cv at the vessel state (info['fallback'] = True).

    P0, T0    vessel pressure [Pa] and gas temperature [K] (stagnation state)
    y         gas-phase composition: dict {name: mole fraction}, or array + names
    Pb        back pressure at the line exit [Pa]
    Cd        orifice contraction/discharge coefficient;  d_orifice [m]
    line      dict(d=diameter, t=wall thickness, L=length) [m], or None
    model     PRMixture containing the components (built and cached if None)
    nozzle    "api520" | "hdi" | "hem"
    expansion "borda" (momentum balance at the orifice-line enlargement) | "none"
    line_id   "d" (diameter = inner diameter) | "d-2t" (diameter = outer diameter)
    info      also return a dict: choke location, P_vc, P1, line pressure drop, ...
    """
    args = (P0, T0, y, Pb, Cd, d_orifice, line, model, names, pseudo, nozzle, expansion,
            line_id, rough, n_iso, n_line, info)
    try:
        return _mdot_core(*args)
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        import valves
        names_, x = _parse(y, names)
        m = model if model is not None else get_model(names_, pseudo)
        if list(m.names) != [n.upper() for n in names_]:
            xx = np.zeros(m.nc)
            for n_, v_ in zip(names_, x):
                xx[m.names.index(n_.upper())] = v_
            x = xx
        st0 = _Vap(m, x, T0, P0)
        ln = None
        if line:
            ln = dict(d=line["d"] if line_id == "d-2t" else line["d"] + 2 * line.get("t", 0.0),
                      t=line.get("t", 0.0), L=line["L"])
        md = valves.mdot_orifice_line(P0, T0, Pb, st0.Z, st0.cp / st0.cv, st0.M, Cd, d_orifice,
                                      ln, mu=m.viscosity(x, T0, st0.rho_mol), rough=rough)
        return (md, dict(fallback=True, mdot=md)) if info else md


def _mdot_core(P0, T0, y, Pb, Cd, d_orifice, line, model, names, pseudo, nozzle, expansion,
               line_id, rough, n_iso, n_line, info):
    names, x = _parse(y, names)
    if model is None:
        model = get_model(names, pseudo)
    if list(model.names) != [n.upper() for n in names]:
        xx = np.zeros(model.nc)
        for n_, v_ in zip(names, x):
            xx[model.names.index(n_.upper())] = v_
        x = xx
    out = dict(choke="none", P_vc=None, P1=None, P_exit=Pb, dP_line=0.0, mdot_orifice=0.0)
    if P0 <= Pb * 1.000001:
        return (0.0, out) if info else 0.0
    A_o = 0.25 * math.pi * d_orifice ** 2
    Pmin = max(Pb, 0.35 * P0)
    st0 = _Vap(model, x, T0, P0)
    if st0.Z < Z_DENSE:
        # dense (near-critical / liquid-like) vessel fluid: the single-phase vapour
        # line integration is not valid (expansion crosses the dome) -> fallback
        raise ValueError("dense vessel state")
    if nozzle == "hem":
        P, v, T = isentrope_hem(model, x, P0, T0, Pmin, n_iso)
    elif nozzle == "hdi":
        P, v, T = isentrope_hdi(model, x, st0, Pmin, n_iso)
    else:
        P, v, T = isentrope_ideal(P0, st0.v, st0.cp / st0.cv, Pmin, n_iso)
    noz = Nozzle(P, v)
    Pstar = max(noz.Pc, Pb)
    md_star = Cd * A_o * noz.Gc
    out["mdot_orifice"] = md_star
    orifice_choked = noz.choked and noz.Pc > Pb
    if not line or line.get("L", 0.0) <= 0.0:
        out["choke"] = "orifice" if orifice_choked else "none"
        out["P_vc"] = Pstar
        return (md_star, out) if info else md_star

    D = line["d"] if line_id == "d" else line["d"] - 2.0 * line.get("t", 0.0)
    D = max(D, d_orifice * 1.0001)
    mu = model.viscosity(x, T0, st0.rho_mol)
    ln = _Line(model, x, st0.h, D, line["L"], mu, rough, n_line)
    borda = expansion == "borda"
    Tg0 = T0 * (Pstar / P0) ** 0.1

    def evaluate(P_vc):
        G, u = noz.state(P_vc) if P_vc > Pstar else (noz.Gc, noz.state(Pstar)[1])
        md = Cd * A_o * G
        Gp = md / ln.A
        P1, s, phi, dTdP = ln.inlet(P_vc, u, Gp, Tg0, borda)
        Lr, ch, Pe = ln.reach(P1, s, phi, dTdP, Gp, Pb) if P1 > Pb else (0.0, False, P1)
        return Lr, ch, Pe, md, P1

    # 0) cheap screen: ideal-gas Fanno (Z0, k0) from P1 = P* without enlargement
    #    recovery (conservative).  If the line needs > 1.5 L to fall to Pb, the
    #    orifice controls with a wide margin and the real-gas line is not integrated.
    if not info and _line_clearly_open(P0, T0, Pstar, Pb, md_star, st0, ln):
        return md_star

    # Line-limited flows: find the vena-contracta pressure P_vc in (P*, P0) at which
    # the line reach equals L.  Residual F = ln(L_reach/L) is smooth and increases
    # with P_vc (less flow).  Written in q = (P_vc - P*)/(P0 - P*).
    key = (round(d_orifice, 6), round(D, 6), round(ln.L, 3), nozzle, expansion, line_id)
    span = P0 - Pstar
    hist = []
    TOL = 3e-4               # in ln(L): mdot error << 1e-4 (d ln L / d ln mdot >> 1)

    def F(q):
        r = evaluate(Pstar + span * q)
        fq = math.log(max(r[0], 1e-9) / ln.L)
        hist.append((q, fq))
        return fq, r

    c = None
    warm = _WARM.get(key)
    if warm is not None and not info:
        # warm start: previous root and slope dF/dq (time stepping -> 1-2 evaluations)
        q, sl = warm
        f0, r0 = F(q)
        for _ in range(6):
            if abs(f0) < TOL:
                c, rc = q, r0
                break
            q1 = q - f0 / sl
            if q1 <= 0.0:
                break                                # orifice may control: full path
            q1 = min(q1, 0.9995)
            f1, r1 = F(q1)
            if f1 != f0 and (f1 - f0) / (q1 - q) > 0:
                sl = (f1 - f0) / (q1 - q)
            q, f0, r0 = q1, f1, r1

    if c is None:
        # 1) orifice-controlled?  (vena contracta at the critical / back pressure)
        Lr, ch, Pe, md, P1 = evaluate(Pstar)
        if Lr >= ln.L or P1 <= Pb:
            out.update(choke="orifice" if orifice_choked else "none", P_vc=Pstar, mdot=md)
            if info:       # line inlet pressure at this flow (lower P_vc, same flow)
                out.update(_orifice_line_info(ln, noz, Cd * A_o, md, Pstar, Pb, Tg0, borda))
            return (md, out) if info else md
        # 2) bracket + Illinois
        a, fa = 0.0, math.log(max(Lr, 1e-9) / ln.L)
        b = warm[0] if warm is not None else 0.3
        fb, rb = F(b)
        while fb < 0.0:                              # need reach > L at the upper end
            a, fa = b, fb
            b = b + 0.5 * (1.0 - b)
            fb, rb = F(b)
            if 1.0 - b < 1e-6:
                break
        c, rc, side = b, rb, 0
        for _ in range(40):
            c = b - fb * (b - a) / (fb - fa)
            fc, rc = F(c)
            if abs(fc) < TOL or abs(b - a) < 1e-9:
                break
            if fc * fb > 0:
                b, fb = c, fc
                if side == -1:
                    fa *= 0.5
                side = -1
            else:
                a, fa = c, fc
                if side == 1:
                    fb *= 0.5
                side = 1
    # store root and local slope for the next call
    hs = sorted(hist, key=lambda t: abs(t[0] - c))[:2]
    sl = (hs[0][1] - hs[1][1]) / (hs[0][0] - hs[1][0]) if len(hs) == 2 and hs[0][0] != hs[1][0] else None
    if sl is None or not sl > 0:
        sl = warm[1] if warm is not None else 5.0
    _WARM[key] = (min(max(c, 1e-4), 0.999), sl)
    c = Pstar + span * c
    Lr, ch, Pe, md, P1 = rc
    out.update(choke="line_exit" if ch else "none", P_vc=c, P1=P1, P_exit=Pe,
               dP_line=P1 - Pe, mdot=md)
    return (md, out) if info else md


def _fanno_F(M, k):
    """Ideal-gas Fanno function f L*/D (Darcy f) from Mach M to choking (Shapiro 1953)."""
    M2 = M * M
    return (1 - M2) / (k * M2) + (k + 1) / (2 * k) * math.log((k + 1) * M2 / (2 + (k - 1) * M2))


def _line_clearly_open(P0, T0, P1, Pb, md, st0, ln, margin=1.5):
    k = st0.cp / st0.cv
    ZRT0 = st0.P * st0.v                                   # Z R T0 per kg
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
    if target <= 1.0:                                      # chokes before reaching Pb
        return _fanno_F(M1, k) * ln.D / f > margin * ln.L
    a, b = M1, 1.0
    for _ in range(50):
        c = 0.5 * (a + b)
        if pr(c) > target:
            a = c
        else:
            b = c
    Mb = 0.5 * (a + b)
    Lb = (_fanno_F(M1, k) - _fanno_F(Mb, k)) * ln.D / f
    return Lb > margin * ln.L


def _orifice_line_info(ln, noz, CdA, md, Pstar, Pb, Tg0, borda):
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
