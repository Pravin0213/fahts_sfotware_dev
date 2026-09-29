"""Blowdown flow through an orifice (BDV) followed by a blowdown line,
real-gas (Peng-Robinson) formulation.  (Ideal-gas fallback: ``fahts.relief.ideal_gas``.)
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
integration failure fall back to fahts.relief.ideal_gas (ideal-gas API 520 + Fanno).
"""

from __future__ import annotations

import math

import numpy as np

from fahts.relief import ideal_gas
from fahts.relief.fanno_line import FannoLine, line_clearly_open, orifice_line_info
from fahts.relief.nozzle import Nozzle, isentrope_hdi, isentrope_hem, isentrope_ideal
from fahts.relief.vapour_state import VapourState
from fahts.thermo import PRMixture


ROUGH = 45e-6
Z_DENSE = 0.5  # below this vessel Z the fallback (ideal-gas) model is used

_MODELS: dict = {}
_WARM: dict = {}


def get_model(names, pseudo=None):
    """PRMixture for a component list (cached)."""
    key = (
        tuple(n.upper() for n in names),
        tuple(sorted((k, tuple(sorted(v.items()))) for k, v in (pseudo or {}).items())),
    )
    m = _MODELS.get(key)
    if m is None:
        m = PRMixture({n: 1.0 for n in names}, pseudo=pseudo)
        _MODELS[key] = m
    return m


def _parse(y, names):
    if isinstance(y, dict):
        names = list(y.keys())
        x = np.array([y[k] for k in names], float)
    else:
        x = np.asarray(y, float)
    x = np.maximum(x, 0.0)
    return list(names), x / x.sum()


def mdot(
    P0,
    T0,
    y,
    Pb,
    Cd,
    d_orifice,
    line=None,
    model=None,
    names=None,
    pseudo=None,
    nozzle="api520",
    expansion="borda",
    line_id="d-2t",
    rough=ROUGH,
    n_iso=12,
    n_line=8,
    info=False,
):
    """Blowdown mass flow [kg/s] (see module docstring).

    Robustness: if the real-gas line/nozzle integration fails (e.g. a near-critical
    dense vessel state), the ideal-gas API 520 + Fanno model of fahts.relief.ideal_gas is used
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
    args = (
        P0,
        T0,
        y,
        Pb,
        Cd,
        d_orifice,
        line,
        model,
        names,
        pseudo,
        nozzle,
        expansion,
        line_id,
        rough,
        n_iso,
        n_line,
        info,
    )
    try:
        return _mdot_core(*args)
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        names_, x = _parse(y, names)
        m = model if model is not None else get_model(names_, pseudo)
        if list(m.names) != [n.upper() for n in names_]:
            xx = np.zeros(m.nc)
            for n_, v_ in zip(names_, x):
                xx[m.names.index(n_.upper())] = v_
            x = xx
        st0 = VapourState(m, x, T0, P0)
        ln = None
        if line:
            ln = dict(
                d=line["d"] if line_id == "d-2t" else line["d"] + 2 * line.get("t", 0.0),
                t=line.get("t", 0.0),
                L=line["L"],
            )
        md = ideal_gas.mdot_orifice_line(
            P0,
            T0,
            Pb,
            st0.Z,
            st0.cp / st0.cv,
            st0.M,
            Cd,
            d_orifice,
            ln,
            mu=m.viscosity(x, T0, st0.rho_mol),
            rough=rough,
        )
        return (md, dict(fallback=True, mdot=md)) if info else md


def _mdot_core(
    P0,
    T0,
    y,
    Pb,
    Cd,
    d_orifice,
    line,
    model,
    names,
    pseudo,
    nozzle,
    expansion,
    line_id,
    rough,
    n_iso,
    n_line,
    info,
):
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
    A_o = 0.25 * math.pi * d_orifice**2
    Pmin = max(Pb, 0.35 * P0)
    st0 = VapourState(model, x, T0, P0)
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
    ln = FannoLine(model, x, st0.h, D, line["L"], mu, rough, n_line)
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
    if not info and line_clearly_open(P0, T0, Pstar, Pb, md_star, st0, ln):
        return md_star

    # Line-limited flows: find the vena-contracta pressure P_vc in (P*, P0) at which
    # the line reach equals L.  Residual F = ln(L_reach/L) is smooth and increases
    # with P_vc (less flow).  Written in q = (P_vc - P*)/(P0 - P*).
    key = (round(d_orifice, 6), round(D, 6), round(ln.L, 3), nozzle, expansion, line_id)
    span = P0 - Pstar
    hist = []
    TOL = 3e-4  # in ln(L): mdot error << 1e-4 (d ln L / d ln mdot >> 1)

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
                break  # orifice may control: full path
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
            if info:  # line inlet pressure at this flow (lower P_vc, same flow)
                out.update(orifice_line_info(ln, noz, Cd * A_o, md, Pstar, Pb, Tg0, borda))
            return (md, out) if info else md
        # 2) bracket + Illinois
        a, fa = 0.0, math.log(max(Lr, 1e-9) / ln.L)
        b = warm[0] if warm is not None else 0.3
        fb, rb = F(b)
        while fb < 0.0:  # need reach > L at the upper end
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
    sl = (
        (hs[0][1] - hs[1][1]) / (hs[0][0] - hs[1][0])
        if len(hs) == 2 and hs[0][0] != hs[1][0]
        else None
    )
    if sl is None or not sl > 0:
        sl = warm[1] if warm is not None else 5.0
    _WARM[key] = (min(max(c, 1e-4), 0.999), sl)
    c = Pstar + span * c
    Lr, ch, Pe, md, P1 = rc
    out.update(
        choke="line_exit" if ch else "none", P_vc=c, P1=P1, P_exit=Pe, dP_line=P1 - Pe, mdot=md
    )
    return (md, out) if info else md
