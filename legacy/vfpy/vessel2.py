"""
vessel2.py - two-phase (vapour / hydrocarbon liquid / free water) vessel model:
fire heat input, blowdown valve (BDV), pressure safety valve (PSV), cold blowdown.

Formulation: TWO-ZONE NON-EQUILIBRIUM (published blowdown practice: Haque et al.
1992; Mahgerefteh & Wong 1999; Speranza & Terenzi 2005; see
studies/13_twophase_physics/DESIGN.txt). All physics from published sources.

Zones
  gas zone  G : moles n_G [mol], enthalpy H_G [J]        (vapour, may form fog)
  liquid    L : moles n_L [mol], enthalpy H_L [J]        (HC liquid and/or free water)
  common pressure P;  V_G + V_L = V_vessel.
Each zone is internally in equilibrium (Peng-Robinson flash, thermo_pr.py); the two
zones exchange heat and mass only through the interface and phase transfer:
  - rain-out: liquid/aqueous phases formed in G are moved to L
  - boil-off / flashing: vapour formed in L is moved to G
  - interface evaporation/condensation from the interface energy balance
    Q_li = Q_ig + mdot * h_fg   (liquid side convection, gas side convection)

Per time step dt (rates explicit from the start-of-step state, wall implicit):
  1. heat rates: dry wall -> gas (natural convection), wet wall -> liquid
     (free convection / nucleate / film boiling, linearised for the implicit wall),
     interface exchange; valve flow from G (valve at the top).
  2. zone enthalpy updates: H_z* = H_z + (Q_z - mdot_out,z h_z) dt (+ interface mass)
  3. pressure: find P with  sum_z N_z v_z(P, h_z(P)) = V,
     h_z(P) = (H_z* + V_z,old (P - P_old)) / N_z   (dH = dQ + V dP for each zone,
     which makes total energy change = Q - H_out exactly since sum V_z = V)
  4. phase transfer between zones at the new P (volume and energy move with the phase)
  5. level -> wetted fraction -> re-weight dry/wet wall columns (energy conserving)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

import twophase_physics as tp
from heat_transfer import (AmbientBC, FireBC, GuidelineFire, Material, WallColumn,
                           VESSFIRE_LOG_NODES_105MM)
from thermo_pr import PRMixture
import valves2
from valves import mdot_orifice_line
from vessel import H_CORRELATIONS, AmbientAuto, PSV, stresses

G_ACC = 9.81
P_ATM = 101325.0
R_GAS = 8.314462618


@dataclass
class Options2:
    flux: str = "blackbody"          # blackbody | balance | none
    eps_surf_fire: float = 0.7
    h_fire: float = 25.0
    h_corr: str = "evans_stefany"    # wall -> gas natural convection. VessFire-matching
                                     # default with the nucleate-only wet wall (study 25);
                                     # "churchill_chu" (horizontal cylinder) is the alternative
    h_corr_liq: str | None = "churchill_chu"   # wall -> single-phase (dense) liquid
    wet_above_crit: str = "boiling"  # wetted-wall heat transfer when the pool cannot
                                     # boil (P above its cricondenbar / latent heat collapsed):
                                     #  "single-phase"  natural convection (physics default)
                                     #  "supercritical" natural convection with integrated cp
                                     #                  (pseudo-critical enhancement; Jackson &
                                     #                  Hall 1979, Pioro & Duffey 2005)
                                     #  "boiling"       keep nucleate-boiling heat transfer with
                                     #                  the pool at saturation (VessFire-like,
                                     #                  see MODEL_CHOICES.md)
    psv_liquid: str = "liquid"       # PSV flow when it draws from the liquid/dense zone:
                                     # "gas" (API 520 vapour equation, real Z, k) or "liquid"
                                     # (API 520 liquid equation, Cd A sqrt(2 rho dP), Kw=Kv=1;
                                     # VessFire-matching: its dense-phase PSV rates equal this
                                     # within 0.3 %, studies/25_top_fixes/dense_relief.csv)
    water_mode: str = "vf"           # free-water pool heat: "physical" (IAPWS pool heats and
                                     # boils), "sink" (pool keeps wall contact, heat it receives
                                     # is removed), "vf" (sink only while the pool holds < 2 kg
                                     # hydrocarbon liquid; VessFire-matching, study 23)
    wet_boiling: str = "nucleate_only" # wetted-wall boiling curve: "full" (nucleate -> CHF ->
                                     # transition -> film; physics) or "nucleate_only" (Cooper
                                     # nucleate boiling, no CHF cap; VessFire-matching: its wet
                                     # wall stays 2-4 K above the liquid all hour, study 22)
    rad_internal: bool = True        # radiation inside the vapour space (dry wall, liquid
                                     # surface, grey gas) - radiosity network
    eps_wall_in: float = 0.8         # inner (oxidised steel) wall emissivity
    eps_liq: float = 0.95            # liquid free-surface emissivity
    eps_gas: float = 0.0             # grey-gas emissivity of the vapour (0 = transparent)
    liq_grashof: str = "drho"        # "beta" (g beta dT) or "drho" (density difference across
                                     # the film, Jackson & Hall 1979) for single-phase liquid
    interface: bool = True           # interface heat exchange
    interface_mass: bool = False     # interface evaporation/condensation (T_i = T_bubble);
                                     # False = sensible exchange, phase change by zone flashes
    boiling: str = "cooper"          # nucleate boiling correlation
    dt: float | None = None
    out_every: float | None = None
    t_end: float | None = None
    sat_every: float = 10.0          # s between saturation-property refreshes
    max_steps_warn: int = 0
    line_diameter: str = "outer"     # blowdown-line diameter input: "outer" (inner = d - 2t,
                                     # default) or "inner" (d is the bore). The manual does not
                                     # say; published line physics gives 6-9 % vs VessFire on the
                                     # LPG cases with d - 2t and 20-37 % with d (study 14).
    use_line: bool = True            # friction in the blowdown line
    valve_model: str = "b1"          # "b1": valves2 (API 520 orifice + Borda-Carnot expansion +
                                     # real-gas PR Fanno line); "simple": valves.py ideal-gas


# ============================================================
# PROPERTY HELPERS  (thermo_pr phases -> correlation property dicts)
# ============================================================

_W = CP.AbstractState("HEOS", "Water")


def _beta(p):
    """Isobaric expansion coefficient from the EOS derivatives of a Phase."""
    return -p.dPdT_v / (p.v * p.dPdv_T)


def phase_dict(m: PRMixture, p) -> dict:
    """rho, cp, mu, k, beta (SI, mass basis) for one Phase."""
    if p.name == "aqueous" or (m.iw is not None and p.x[m.iw] > 0.99):
        try:
            _W.update(CP.PT_INPUTS, p.P, p.T)
        except ValueError:                           # exactly saturated: liquid side
            _W.update(CP.PQ_INPUTS, p.P, 0.0 if p.name == "aqueous" else 1.0)
        return dict(rho=_W.rhomass(), cp=_W.cpmass(), mu=_W.viscosity(), k=_W.conductivity(),
                    beta=_W.isobaric_expansion_coefficient())
    rho_t = m.transport_rho(p)
    return dict(rho=p.rho, cp=p.cp_mass, mu=m.viscosity(p.x, p.T, rho_t),
                k=m.thermal_conductivity(p.x, p.T, rho_t), beta=_beta(p))


def film_gas(m: PRMixture, y, T, P) -> dict:
    return phase_dict(m, m.phase_props(y, T, P, "V"))


def sat_props(m: PRMixture, x, P, T_guess=None, water=False) -> dict:
    """Saturation property dict for twophase_physics correlations (bubble point of
    liquid composition x at P; pure-water properties for a free-water pool)."""
    if water:
        _W.update(CP.PQ_INPUTS, P, 0.0)
        s = dict(P=P, P_c=_W.p_critical(), M=18.015, T_sat=_W.T(), rho_l=_W.rhomass(),
                 h_l=_W.hmass(), mu_l=_W.viscosity(), k_l=_W.conductivity(), cp_l=_W.cpmass(),
                 sigma=_W.surface_tension())
        _W.update(CP.PQ_INPUTS, P, 1.0)
        s.update(rho_v=_W.rhomass(), h_v=_W.hmass(), mu_v=_W.viscosity(), k_v=_W.conductivity(),
                 cp_v=_W.cpmass())
        s["h_fg"] = s["h_v"] - s["h_l"]
        return s
    Tb, y = m.bubble_T(P, x, T_guess)
    L = m.phase_props(x, Tb, P, "L")
    V = m.phase_props(y, Tb, P, "V")
    dl, dv = phase_dict(m, L), phase_dict(m, V)
    sig = m.surface_tension(x, m.transport_rho(L), y, m.transport_rho(V))
    x = np.asarray(x) / np.sum(x)
    # Kay's-rule pseudo-critical pressure underestimates a mixture's true critical
    # pressure (often below the operating P for rich liquids), which would make the
    # reduced-pressure boiling correlations invalid (log of p_r > 1); cap p_r at 0.9,
    # the upper end of Cooper's (1984) data range.
    return dict(P=P, P_c=max(float(x @ m.Pc), P / 0.9), M=L.M * 1e3, T_sat=Tb,
                rho_l=L.rho, h_l=L.h_mass, mu_l=dl["mu"], k_l=dl["k"], cp_l=dl["cp"],
                rho_v=V.rho, h_v=V.h_mass, mu_v=dv["mu"], k_v=dv["k"], cp_v=dv["cp"],
                sigma=max(sig, 1e-5), h_fg=max(V.h_mass - L.h_mass, 1e3), y=y)


def nucleate_only_flux(T_w, T_l, sat, liq_props, L_nc, n_blend=3.0):
    """Wall -> pool heat flux on the nucleate-boiling branch only (Cooper 1984, blended
    with free convection, Churchill-Usagi), no critical-heat-flux limit. Cooper needs
    only reduced pressure, molar mass and q, so it stays defined near the critical
    point where h_fg and sigma -> 0 (and CHF would collapse)."""
    q_nc = tp.q_free(T_w, T_l, liq_props, L_nc, "vertical_plate")
    dT = T_w - sat["T_sat"]
    if dT <= 0.0:
        return q_nc
    q_nb = tp.nb_cooper(sat)(dT)
    return (max(q_nc, 0.0) ** n_blend + q_nb ** n_blend) ** (1.0 / n_blend)


def rad_network(T1, T3, Tg, A1, A3, e1, e3, eg):
    """Grey radiation exchange in the vapour space (radiosity network; e.g. Modest,
    Radiative Heat Transfer, ch. 4 and 19; Incropera ch. 13).
    Surfaces: 1 = dry wall inner surface (area A1), 3 = liquid free surface (A3, flat,
    so F31 = 1 and F13 = A3/A1, F11 = 1 - F13); grey gas of emissivity eg (transmissivity
    1 - eg over the enclosure's mean beam length).
    Returns (Q1, Q3, Qg): net radiation LEAVING surface 1, LEAVING surface 3, and
    ABSORBED by the gas (Q1 + Q3 = Qg), in W."""
    S = 5.670374419e-8
    E1, E3, Eg = S * T1**4, S * T3**4, S * Tg**4
    tau = 1.0 - eg
    if A1 <= 0:
        return 0.0, 0.0, 0.0
    G1 = e1 * A1 / (1.0 - e1) if e1 < 1 else 1e12 * A1
    G1g = A1 * eg                                   # surface 1 <-> gas (F1,all = 1)
    if A3 <= 0:
        if eg <= 0:
            return 0.0, 0.0, 0.0
        J1 = (G1 * E1 + G1g * Eg) / (G1 + G1g)
        Q1 = G1 * (E1 - J1)
        return Q1, 0.0, Q1
    G3 = e3 * A3 / (1.0 - e3) if e3 < 1 else 1e12 * A3
    G13 = A3 * tau                                  # A1 F13 tau = A3 F31 tau
    G3g = A3 * eg
    # node equations: G1(E1-J1) + G13(J3-J1) + G1g(Eg-J1) = 0 ; same for J3
    a11, a12, b1 = G1 + G13 + G1g, -G13, G1 * E1 + G1g * Eg
    a21, a22, b2 = -G13, G3 + G13 + G3g, G3 * E3 + G3g * Eg
    det = a11 * a22 - a12 * a21
    J1 = (b1 * a22 - a12 * b2) / det
    J3 = (a11 * b2 - a21 * b1) / det
    Q1, Q3 = G1 * (E1 - J1), G3 * (E3 - J3)
    return Q1, Q3, Q1 + Q3


def api520_G(P0, T0, Pb, Z, k, M):
    """Ideal (Kd = 1) gas mass flux [kg/m2/s], API 520 Part I, real Z and k."""
    if P0 <= Pb:
        return 0.0
    R = R_GAS / M
    r = Pb / P0
    rc = (2 / (k + 1)) ** (k / (k - 1))
    if r <= rc:
        return math.sqrt(k * (2 / (k + 1)) ** ((k + 1) / (k - 1))) * P0 / math.sqrt(Z * R * T0)
    F2 = math.sqrt((k / (k - 1)) * r ** (2 / k) * (1 - r ** ((k - 1) / k)) / (1 - r))
    return F2 * math.sqrt(2 * (P0 - Pb) * P0 / (Z * R * T0))


def fast_PH_single(m, x, P, h, T0, root):
    """Single-phase isenthalpic state (Newton on T with the given EOS root 'V'/'L').
    Used inside the pressure iteration; phase stability is checked once afterwards."""
    from thermo_pr import FlashResult
    T, ok = T0, False
    try:
        for _ in range(12):
            p = m.phase_props(x, T, P, root)
            if not (np.isfinite(p.cp) and p.cp > 0):
                return None
            dT = (h - p.h) / p.cp
            dT = max(min(dT, 20.0), -20.0)
            T = min(max(T + dT, 0.7 * T0), 1.5 * T0)
            if abs(dT) < 1e-7 * T:
                ok = True
                break
        if not ok:
            return None
        p = m.phase_props(x, T, P, root)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    if abs(p.h - h) > 1e-4 * max(abs(h), 1.0) + 0.5:
        return None
    p.beta = 1.0
    return FlashResult(T, P, np.asarray(x, float), [p], kind="fast1", names=m.names)


def fast_PH_multi(m, x, P, h, prev):
    """Isenthalpic flash assuming the phase set of `prev` (no stability search):
    Newton on T with flash_PT(same_phases=True). Returns None if the phase set is no
    longer valid (a phase fraction left (0, 1)) so the caller can do a full flash."""
    T = prev.T
    r = m.flash_PT(P, T, x, init=prev, same_phases=True)
    for _ in range(12):
        dT_fd = 0.01
        r2 = m.flash_PT(P, T + dT_fd, x, init=r, same_phases=True)
        cp_eff = (r2.h - r.h) / dT_fd
        if not np.isfinite(cp_eff) or cp_eff <= 0:
            return None
        dT = (h - r.h) / cp_eff
        dT = max(min(dT, 20.0), -20.0)
        T += dT
        r = m.flash_PT(P, T, x, init=r, same_phases=True)
        if abs(dT) < 1e-6:
            break
    if abs(r.h - h) > 1e-3 * max(abs(h), 1.0) + 1.0:
        return None
    if any(not (0.0 < p.beta < 1.0) for p in r.phases) or len(r.phases) != len(prev.phases):
        return None
    return r


# ============================================================
# ZONE
# ============================================================

class Zone:
    """Moles n [mol], enthalpy H [J] and the last flash result r at pressure P."""

    def __init__(self, name, n, H, r):
        self.name, self.n, self.H, self.r = name, np.asarray(n, float), float(H), r

    @property
    def N(self):
        return float(self.n.sum())

    @property
    def empty(self):
        return self.N < 1e-3

    @property
    def V(self):
        return self.N * self.r.v if not self.empty else 0.0

    @property
    def mass(self):
        return self.N * self.r.M if not self.empty else 0.0

    @property
    def T(self):
        return self.r.T


def split_phases(r, N, keep):
    """Split a zone flash result into (kept phases, moved phases): each as
    (moles vector, enthalpy J, volume m3). keep = tuple of phase names to keep."""
    kn, kH, kV = np.zeros(len(r.z)), 0.0, 0.0
    mn, mH, mV = np.zeros(len(r.z)), 0.0, 0.0
    for p in r.phases:
        nn = N * p.beta * np.asarray(p.x)
        if p.name in keep:
            kn += nn; kH += N * p.beta * p.h; kV += N * p.beta * p.v
        else:
            mn += nn; mH += N * p.beta * p.h; mV += N * p.beta * p.v
    return (kn, kH, kV), (mn, mH, mV)


# ============================================================
# SIMULATION
# ============================================================

def simulate2(case: dict, opt: Options2) -> tuple[pd.DataFrame, dict]:
    s, hl, adm = case["seg"], case["hl"], case["admin"]
    fl = dict(s["fluid"])
    pseudo = {k: dict(sg=v["rd"], Tb=v["Tb"]) for k, v in s.get("pseudo", {}).items()}
    m = PRMixture(fl, pseudo=pseudo or None)
    names = m.names
    z0 = np.array([fl[n] if n in fl else fl.get(n.upper(), 0.0) for n in names], float)
    z0 /= z0.sum()

    mat = Material.from_vessfire_db(s["material"])
    D, t_w, L = s["D"], s["t"], s["L"]
    D_out = D + 2 * t_w
    geom = tp.VesselGeometry(D=D, L=L, orientation="horizontal", head="flat")
    V = geom.V_total
    A_in, A_out = math.pi * D * L, math.pi * D_out * L
    ser = hl["series"]
    fire_on = opt.flux != "none" and np.any(ser[:, 1:] > 0)

    def outer_bc():
        amb = AmbientAuto(s["T_env"], s["eps_surf"], D_out, s["h_out"])
        if not fire_on:
            return amb
        fire = GuidelineFire(eps_flame=1.0, eps_surf=opt.eps_surf_fire, h_flame=opt.h_fire,
                             T_ref=s["T_shell"],
                             t_air_mode="blackbody" if opt.flux == "blackbody" else "balance")
        return FireBC(fire, ser[:, 0], ser[:, 1] * 1e3, after=amb)

    # ---------------------------------------------------------------- initial state
    P, T0 = s["P0"], s["T0"]
    r0 = m.flash_PT(P, T0, z0)
    V_w = geom.volume(s.get("water_level", 0.0)) if s.get("water_level", 0.0) > 0 else 0.0
    V_hc_tot = geom.volume(s.get("hc_level", 0.0) + s.get("water_level", 0.0)) \
        if s.get("hc_level", 0.0) > 0 else V_w
    V_oil = max(V_hc_tot - V_w, 0.0)
    Vp, Lp, Wp = r0.vapour, r0.liquid, r0.aqueous
    if Lp is None:
        V_oil = 0.0
    if Wp is None:
        V_w = 0.0
    V_gas = V - V_oil - V_w
    nG = (V_gas / Vp.v) * np.asarray(Vp.x) if Vp is not None else np.zeros(len(names))
    HG = (V_gas / Vp.v) * Vp.h if Vp is not None else 0.0
    nL, HL = np.zeros(len(names)), 0.0
    if Lp is not None and V_oil > 0:
        nL += (V_oil / Lp.v) * np.asarray(Lp.x); HL += (V_oil / Lp.v) * Lp.h
    if Wp is not None and V_w > 0:
        nL += (V_w / Wp.v) * np.asarray(Wp.x); HL += (V_w / Wp.v) * Wp.h
    if Vp is None:
        raise ValueError("initial state has no vapour phase - liquid-full vessels not supported")
    G = Zone("gas", nG, HG, m.flash_PH(P, HG / nG.sum(), nG / nG.sum(), T_guess=T0))
    Lz = Zone("liq", nL, HL, m.flash_PH(P, HL / nL.sum(), nL / nL.sum(), T_guess=T0)) \
        if nL.sum() > 1e-3 else Zone("liq", nL, 0.0, r0)

    # ---------------------------------------------------------------- wall
    def lvl():
        return geom.level(Lz.V) if not Lz.empty else 0.0

    f_wet = geom.wetted_perimeter_fraction(lvl()) if not Lz.empty else 0.0
    cols = {"dry": WallColumn(mat, D / 2, VESSFIRE_LOG_NODES_105MM, outer_bc(), s["T_shell"]),
            "wet": WallColumn(mat, D / 2, VESSFIRE_LOG_NODES_105MM, outer_bc(), s["T_shell"])}
    frac = {"dry": 1.0 - f_wet, "wet": f_wet}
    T_TAB = np.linspace(50.0, 2000.0, 7801)
    E_TAB = np.concatenate(([0.0], np.cumsum(0.5 * (mat.cp_at(T_TAB[1:]) + mat.cp_at(T_TAB[:-1]))
                                              * np.diff(T_TAB))))

    def e_of_T(T):
        return np.interp(T, T_TAB, E_TAB)

    # ---------------------------------------------------------------- valves
    Pb = s["back_pressure"]
    bdv = s.get("bdv")
    A_bdv = math.pi / 4 * bdv["d"] ** 2 if bdv else 0.0
    psv = PSV(None, s["psv"], "api520") if s.get("psv") else None
    A_psv = math.pi / 4 * s["psv"]["d"] ** 2 if s.get("psv") else 0.0

    dt = opt.dt or float(adm.get("max_timestep", 1.0))
    t_end = opt.t_end or float(adm["maxtime"])
    out_every = opt.out_every or float(adm.get("output_frequence", 10))
    n_steps = int(round(t_end / dt))

    E_wall0 = sum(c.energy() * frac[k] for k, c in cols.items()) * L
    m_liq0 = Lz.mass if not Lz.empty else 0.0
    m_gas0 = G.mass
    U0 = (G.H - P * G.V) + ((Lz.H - P * Lz.V) if not Lz.empty else 0.0)
    Q_fire = H_out = 0.0
    Q_sink = [0.0]
    thr = {"S": 0.0}          # heat + enthalpy throughput, scale for the energy error
    sticky = {"G": False, "L": False}
    last_split = {"G": None, "L": None}
    released = 0.0
    rows = []
    rupture = {"vonMises": None, "Tresca": None}
    sat = {"t": -1e9, "d": None}
    info = dict(h_dry=0.0, h_wet=0.0, regime="", Q_wg=0.0, Q_wl=0.0, Q_ig=0.0, Q_li=0.0,
                m_evap=0.0, md_bdv=0.0, md_psv=0.0)

    def gas_vapour():
        if G.empty and not Lz.empty:
            return Lz.r.phases[0]
        return G.r.vapour or G.r.phases[0]

    def record(time):
        hot = max(cols, key=lambda k: cols[k].T_mean() if frac[k] > 0 else -1)
        Tm = cols[hot].T_mean()
        f = mat.f_uts_at(Tm) if s.get("stress_type", "U") == "U" else mat.f_yield_at(Tm)
        allow = s["strength_mpa"] * s.get("stress_factor", 1.0) * f
        s_h, s_l, vm, tr = stresses(P, D, t_w, s.get("ext_long_mpa", 0.0))
        for crit, val in (("vonMises", vm), ("Tresca", tr)):
            if rupture[crit] is None and val >= allow:
                rupture[crit] = time
        E_wall = sum(c.energy() * frac[k] for k, c in cols.items()) * L
        U = (G.H - P * G.V) + ((Lz.H - P * Lz.V) if not Lz.empty else 0.0)
        bal = (E_wall - E_wall0) + (U - U0) + H_out + Q_sink[0] - Q_fire
        gv = gas_vapour()
        r = dict(Time=time, P_bara=P / 1e5, P_barg=(P - P_ATM) / 1e5,
                 T_gas_C=G.T - 273.15, T_liq_C=(Lz.T - 273.15) if not Lz.empty else np.nan,
                 m_gas=G.mass, m_gas_zone=G.mass, m_liq=Lz.mass, level=lvl(), f_wet=frac["wet"],
                 MW_gas=gv.M * 1e3, mdot_bdv=info["md_bdv"], mdot_psv=info["md_psv"],
                 released=released, sigma_vM=vm, sigma_Tresca=tr, sigma_allow=allow,
                 T_mean_hot_C=Tm - 273.15, h_dry=info["h_dry"], h_wet=info["h_wet"],
                 regime=info["regime"], Q_wg_kW=info["Q_wg"] / 1e3, Q_wl_kW=info["Q_wl"] / 1e3,
                 Q_ig_kW=info["Q_ig"] / 1e3, Q_li_kW=info["Q_li"] / 1e3, m_evap=info["m_evap"],
                 Q_rad_liq_kW=info.get("Q_rad_l", 0.0) / 1e3, Q_rad_gas_kW=info.get("Q_rad_g", 0.0) / 1e3,
                 h_rad=info.get("h_rad", 0.0),
                 energy_err_MJ=bal / 1e6, Q_fire_cum_MJ=Q_fire / 1e6, Q_sink_MJ=Q_sink[0] / 1e6,
                 energy_err_pct=100 * bal / max(thr["S"], 1.0))
        for k, c in cols.items():
            key = "background" if k == "dry" else "wet"
            r[f"{key}_T_out_C"] = c.T_outer - 273.15
            r[f"{key}_T_in_C"] = c.T_inner - 273.15
            r[f"{key}_T_mean_C"] = c.T_mean() - 273.15
            r[f"{key}_q_net_kW"] = c.q_net_out / 1e3
            for i, Ti in enumerate(c.T):
                r[f"{key}_T{i+1}_C"] = Ti - 273.15
        rows.append(r)

    for c in cols.values():
        c.q_net_out, _, c.q_rad_out, c.q_conv_out = c.outer(c.T_outer, 0.0)
    record(0.0)
    next_out = out_every

    for step in range(1, n_steps + 1):
        time = step * dt
        t_mid = time - 0.5 * dt
        gv = gas_vapour()
        yG = np.asarray(gv.x)
        src_is_gas = not G.empty
        T_src = G.T if src_is_gas else Lz.T

        # ---------------- saturation properties of the liquid pool (refreshed)
        liq_present = not Lz.empty
        if liq_present and time - sat["t"] >= opt.sat_every:
            xL = Lz.n / Lz.N
            water_pool = m.iw is not None and xL[m.iw] > 0.99
            # bubble point: continuation from the last T_sat first (the PR saturation solver
            # fails from a guess well below T_sat near the cricondenbar), then the liquid
            # temperature, then the Wilson estimate (study 15)
            sd_new = None
            for g_ in ([sat.get("Tsat")] if sat.get("Tsat") else []) + [Lz.T, None]:
                try:
                    sd_new = sat_props(m, xL, P, T_guess=g_, water=water_pool)
                    break
                except Exception:
                    sd_new = None                  # no bubble point at this P (supercritical)
            if sd_new is not None:
                sat["Tsat"] = sd_new["T_sat"]
                # no genuine boiling when the pool is near/above its critical region
                # (latent heat collapsing): treat the wetted wall as single-phase
                if not water_pool and (sd_new["h_fg"] < 30e3 or sd_new["T_sat"] < Lz.T - 50.0):
                    sd_new = None
            sat["d"], sat["t"] = sd_new, time
            if sd_new is not None:
                sat["last"] = sd_new
        sd = sat["d"]

        # ---------------- valves (gas zone, top-mounted)
        Z, kk, Mv = gv.Z, gv.cp / gv.cv, gv.M
        if bdv and t_mid >= bdv["delay"]:
            line = s.get("bdv_line") if opt.use_line else None
            if opt.valve_model == "b1":
                md_bdv = valves2.mdot(P, T_src, yG, Pb, bdv["cd"], bdv["d"], line, model=m,
                                      names=names, line_id="d" if opt.line_diameter == "inner" else "d-2t")
            else:
                if line and opt.line_diameter == "inner":
                    line = dict(line, t=0.0)
                md_bdv = mdot_orifice_line(P, G.T, Pb, Z, kk, Mv, bdv["cd"], bdv["d"], line,
                                           mu=sat.get("pg", {}).get("mu", 1.2e-5))
        else:
            md_bdv = 0.0
        if psv and not src_is_gas and opt.psv_liquid == "liquid":
            md_psv = A_psv * s["psv"]["cd"] * math.sqrt(2.0 * gv.rho * max(P - Pb, 0.0)) * psv.frac(P)
        else:
            md_psv = A_psv * s["psv"]["cd"] * api520_G(P, T_src, Pb, Z, kk, Mv) * psv.frac(P) if psv else 0.0
        mdot = md_bdv + md_psv
        info["md_bdv"], info["md_psv"] = md_bdv, md_psv

        # ---------------- dry wall -> gas
        cd = cols["dry"]
        pg = film_gas(m, yG, 0.5 * (cd.T_inner + G.T), P)
        sat["pg"] = pg
        dTg = abs(cd.T_inner - G.T)
        if dTg > 1e-3:
            Pr = pg["cp"] * pg["mu"] / pg["k"]
            Ra = G_ACC * abs(pg["beta"]) * dTg * D**3 * pg["rho"]**2 * pg["cp"] / (pg["mu"] * pg["k"])
            h_dry = H_CORRELATIONS[opt.h_corr](Ra, Pr) * pg["k"] / D
        else:
            h_dry = 0.0
        T_fl = G.T if src_is_gas else Lz.T
        Q_rad_l = Q_rad_g = 0.0
        if opt.rad_internal and frac["dry"] > 0 and src_is_gas:
            A1 = A_in * frac["dry"]
            A3 = geom.interface_area(geom.level(Lz.V)) if not Lz.empty else 0.0
            T3 = Lz.T if not Lz.empty else G.T
            T1 = cd.T_inner
            Q1, Q3, Qg = rad_network(T1, T3, G.T, A1, A3, opt.eps_wall_in, opt.eps_liq, opt.eps_gas)
            Q1b = rad_network(T1 + 0.05, T3, G.T, A1, A3, opt.eps_wall_in, opt.eps_liq, opt.eps_gas)[0]
            h_r = max((Q1b - Q1) / 0.05 / A1, 1e-6)             # W/m2K, radiative conductance
            T_ref_r = T1 - (Q1 / A1) / h_r
            h_tot = h_dry + h_r
            T_eff = (h_dry * T_fl + h_r * T_ref_r) / h_tot
            cd.step(dt, t_mid, T_eff, h_tot)
            Q1_new = h_r * (cd.T_inner - T_ref_r) * A1          # radiation actually leaving the wall
            # the liquid surface receives its network share (scaled to the implicit wall
            # value); the gas absorbs the remainder, so energy is conserved exactly
            scale = Q1_new / Q1 if abs(Q1) > 1.0 else 1.0
            Q_rad_l = -Q3 * scale
            Q_rad_g = Q1_new - Q_rad_l
            Q_wg = h_dry * (cd.T_inner - T_fl) * A1 + Q_rad_g
            info["h_rad"] = h_r
        else:
            q = cd.step(dt, t_mid, T_fl, h_dry)
            Q_wg = q * A_in * frac["dry"]
        Q_fire += cd.q_net_out * A_out * frac["dry"] * dt
        info["h_dry"] = h_dry
        info["Q_rad_l"], info["Q_rad_g"] = Q_rad_l, Q_rad_g

        # ---------------- wet wall -> liquid
        cw = cols["wet"]
        Q_wl = 0.0
        if (liq_present and sd is None and opt.wet_above_crit == "boiling"
                and sat.get("last") is not None and not (m.iw is not None and (Lz.n / Lz.N)[m.iw] > 0.99)):
            # VessFire-like: boiling-level heat transfer continues with the pool taken to be
            # at saturation, using the last valid saturation property set
            sd = dict(sat["last"], T_sat=Lz.T, P=P)
        if liq_present and frac["wet"] > 0 and sd is None:
            # single-phase natural convection to a dense (compressed / supercritical) pool
            Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
            # film-temperature properties (as on the gas side): for a dense / near-critical
            # pool the properties change strongly between bulk and wall temperature
            T_film = 0.5 * (cw.T_inner + Lz.T)
            try:
                pl = phase_dict(m, m.phase_props(Lph.x, T_film, P, "L"))                     if Lph.name != "aqueous" else phase_dict(m, Lph)
            except Exception:
                pl = phase_dict(m, Lph)
            Lnc = geom.liquid_char_length(geom.level(Lz.V))
            dTl = abs(cw.T_inner - Lz.T)
            if dTl > 1e-3:
                Pr_l = pl["cp"] * pl["mu"] / pl["k"]
                if opt.liq_grashof == "drho" and Lph.name != "aqueous":
                    # density-difference buoyancy for large property variation across the
                    # film (supercritical / near-critical fluids; Jackson & Hall 1979):
                    # Ra = g (rho_b - rho_w) L^3 rho_f cp_f / (mu_f k_f)
                    try:
                        rho_w = m.phase_props(Lph.x, cw.T_inner, P, "L").rho
                    except Exception:
                        rho_w = Lph.rho
                    drho = abs(Lph.rho - rho_w)
                    cp_use = pl["cp"]
                    if opt.wet_above_crit == "supercritical":
                        # integrated (bulk-to-wall) mean cp captures the pseudo-critical peak
                        try:
                            hw = m.phase_props(Lph.x, cw.T_inner, P, "L").h_mass
                            cp_bar = (hw - Lph.h_mass) / (cw.T_inner - Lz.T)
                            if np.isfinite(cp_bar) and cp_bar > 0:
                                cp_use = cp_bar
                        except Exception:
                            pass
                        Pr_l = cp_use * pl["mu"] / pl["k"]
                    Ra_l = G_ACC * drho * Lnc**3 * pl["rho"] * cp_use / (pl["mu"] * pl["k"])
                else:
                    Ra_l = G_ACC * abs(pl["beta"]) * dTl * Lnc**3 * pl["rho"]**2 * pl["cp"] / (pl["mu"] * pl["k"])
                h_w = H_CORRELATIONS[opt.h_corr_liq or opt.h_corr](Ra_l, Pr_l) * pl["k"] / Lnc
            else:
                h_w = 0.0
            cw.step(dt, t_mid, Lz.T, h_w)
            Q_wl = h_w * (cw.T_inner - Lz.T) * A_in * frac["wet"]
            info["h_wet"], info["regime"] = h_w, "single-phase"
        elif liq_present and frac["wet"] > 0 and sd is not None:
            Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
            pl = phase_dict(m, Lph)
            h_l = geom.level(Lz.V)
            Lnc = geom.liquid_char_length(h_l)
            try:
                if opt.wet_boiling == "nucleate_only":
                    qfun = lambda Tw: nucleate_only_flux(Tw, Lz.T, sd, pl, Lnc)
                    q0, dq = tp.with_derivative(qfun, cw.T_inner)
                    reg = "nucleate_only"
                else:
                    q0, dq, reg = tp.boiling_flux_and_derivative(cw.T_inner, Lz.T, sd, pl, Lnc)
            except Exception:
                q0, dq, reg = 0.0, 1.0, "fail"
            h_eff, T_ref = tp.linearised_bc(q0, dq, cw.T_inner)
            cw.step(dt, t_mid, T_ref, h_eff)
            Q_wl = h_eff * (cw.T_inner - T_ref) * A_in * frac["wet"]
            info["h_wet"], info["regime"] = h_eff, reg
        else:
            cw.step(dt, t_mid, G.T, h_dry)               # keeps the (zero-area) column alive
        Q_fire += cw.q_net_out * A_out * frac["wet"] * dt

        # ---------------- interface
        Q_ig = Q_li = m_ev = 0.0
        n_ev = np.zeros(len(names)); H_evL = H_evG = 0.0
        if opt.interface and liq_present and not G.empty and (sd is not None or not opt.interface_mass):
            h_l = geom.level(Lz.V)
            Ai, Ls = geom.interface_area(h_l), geom.interface_char_length(h_l)
            Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
            xLz = Lz.n / Lz.N
            if not opt.interface_mass or (m.iw is not None and xLz[m.iw] > 0.99):
                # Sensible exchange only (default; always for a free-water pool): gas- and
                # liquid-side free convection in series, interface temperature from the
                # surface energy balance, no interface mass transfer. Phase change then
                # happens through each zone's own equilibrium flash (boil-off when the
                # liquid exceeds its bubble point, rain-out when the gas is supersaturated),
                # as in the BLOWDOWN formulation (Haque et al. 1992). This avoids the
                # (Q_li - Q_ig)/h_fg interface rate, which diverges near the mixture
                # critical point where h_fg -> 0. For water under a non-condensable gas the
                # surface is also not at T_sat(P) (water partial pressure is tiny).
                pgi, pli = film_gas(m, yG, G.T, P), phase_dict(m, Lph)
                Ti = 0.5 * (G.T + Lz.T)
                for _ in range(20):
                    hg = tp.h_free(pgi, G.T - Ti, Ls, "hot_down" if G.T > Ti else "hot_up")
                    hl = tp.h_free(pli, Lz.T - Ti, Ls, "hot_up" if Lz.T > Ti else "hot_down")
                    Ti_new = (hg * G.T + hl * Lz.T) / max(hg + hl, 1e-9)
                    if abs(Ti_new - Ti) < 1e-4:
                        break
                    Ti = Ti_new
                Qs = hl * Ai * (Lz.T - Ti)             # liquid -> interface = interface -> gas
                ie = dict(Q_ig=Qs, Q_li=Qs, mdot_evap=0.0)
            else:
                ie = tp.interface_exchange(G.T, Lz.T, sd, film_gas(m, yG, G.T, P), phase_dict(m, Lph), Ai, Ls)
            Q_ig, Q_li, m_ev = ie["Q_ig"], ie["Q_li"], ie["mdot_evap"] * dt
            # limit phase change to what the zones can supply in a step
            if m_ev > 0:
                m_ev = min(m_ev, 0.2 * Lz.mass)
                ycomp = np.asarray(sd.get("y", yG)); Mev = float(ycomp @ m.Mw)
                n_ev = m_ev / Mev * ycomp                 # evaporating (incipient vapour) moles
                n_ev = np.minimum(n_ev, 0.5 * Lz.n)
                H_evL, H_evG = m_ev * sd["h_l"], m_ev * sd["h_v"]
            elif m_ev < 0:
                m_ev = max(m_ev, -0.2 * G.mass)
                n_ev = m_ev / (yG @ m.Mw) * yG             # condensing gas moles (negative)
                H_evL, H_evG = m_ev * sd["h_l"], m_ev * sd["h_v"]
        info.update(Q_wg=Q_wg, Q_wl=Q_wl, Q_ig=Q_ig, Q_li=Q_li, m_evap=m_ev / dt)
        thr["S"] += (abs(Q_wg) + abs(Q_wl) + abs(Q_ig) + abs(Q_li) + abs(Q_rad_l)) * dt             + abs(mdot * dt * gv.h / gv.M)

        # ---------------- zone contents after flows (enthalpy, before pressure update)
        n_out = mdot * dt / Mv * yG
        n_out = np.minimum(n_out, 0.5 * (G.n if src_is_gas else Lz.n))
        H_o = float(n_out.sum()) * gv.h
        H_out += H_o
        released += mdot * dt
        G_n = G.n - (n_out if src_is_gas else 0.0) + n_ev
        G_H = G.H + (Q_wg + Q_ig) * dt - (H_o if src_is_gas else 0.0) + H_evG
        L_n = Lz.n - n_ev - (0.0 if src_is_gas else n_out)
        sink = False
        if opt.water_mode == "sink":
            sink = True
        elif opt.water_mode == "vf" and m.iw is not None and liq_present:
            m_hc = float(np.delete(Lz.n, m.iw) @ np.delete(m.Mw, m.iw))
            sink = Lz.n[m.iw] > 0 and m_hc < 2.0
        Q_liq_in = Q_wl - Q_li + Q_rad_l
        if sink:
            Q_sink[0] += Q_liq_in * dt
            Q_liq_in = 0.0
        L_H = Lz.H + Q_liq_in * dt - H_evL - (0.0 if src_is_gas else H_o)
        VG0, VL0, P0s = G.V, Lz.V, P

        # ---------------- pressure: volumes of both zones fill the vessel
        # Zones that were single-phase last step use a fast single-phase solve during
        # the iteration; one stability test afterwards switches a zone to a full
        # flash (and re-solves P) only if a new phase has appeared.
        def zone_flash(n, H, Vold, Pt, prev, full, root, split_prev=None):
            N = n.sum()
            if N < 1e-3:
                return None, 0.0
            h = (H + Vold * (Pt - P0s)) / N
            x = n / N
            if m.iw is not None and x[m.iw] >= 1.0 - 1e-12:
                rw = m.water_PH(Pt, h, T_guess=prev.T if prev is not None else None)
                return rw, N * rw.v
            r = None
            if not full and prev is not None and len(prev.phases) == 1                     and prev.phases[0].name != "aqueous":       # water pools use IAPWS (full flash)
                r = fast_PH_single(m, x, Pt, h, prev.T, root)
            elif prev is not None and len(prev.phases) > 1:
                r = fast_PH_multi(m, x, Pt, h, prev)          # same phase set, no stability
            elif full and split_prev is not None:
                r = fast_PH_multi(m, x, Pt, h, split_prev)    # zone split last step too
            if r is None:
                r = m.flash_PH(Pt, h, x, T_guess=prev.T if prev is not None else 300.0, init=prev)
            return r, N * r.v

        full = dict(sticky)          # zones that needed a full flash last step start full
        prevL = Lz.r if not Lz.empty else None
        cache = {}

        def resid(Pt):
            rg, vg = zone_flash(G_n, G_H, VG0, Pt, G.r, full["G"], "V", last_split["G"])
            rl, vl = zone_flash(L_n, L_H, VL0, Pt, prevL, full["L"], "L", last_split["L"])                 if L_n.sum() > 1e-3 else (None, 0.0)
            cache[Pt] = (rg, rl)
            return vg + vl - V

        def solve_P(P_start):
            Pa, fa = P_start, resid(P_start)
            slope = sat.get("dVdP")
            Pb_ = Pa - fa / slope if slope else (Pa * (1.0 - 1e-4) if fa < 0 else Pa * (1.0 + 1e-4))
            if not (0.5 * Pa < Pb_ < 2.0 * Pa):
                Pb_ = Pa * (1.0 - 1e-4) if fa < 0 else Pa * (1.0 + 1e-4)
            def safe(Pt, P_good):
                # a trial pressure far from the solution can put a zone where no flash
                # solution exists; back off towards the last good pressure
                for _ in range(12):
                    try:
                        return Pt, resid(Pt)
                    except (RuntimeError, ValueError, ZeroDivisionError, FloatingPointError):
                        Pt = 0.5 * (Pt + P_good)
                return Pt, resid(Pt)
            Pb_, fb = safe(Pb_, Pa)
            for _ in range(40):
                if abs(fb) < 1e-8 * V or fb == fa:
                    break
                Pn = Pb_ - fb * (Pb_ - Pa) / (fb - fa)
                Pn = min(max(Pn, 0.8 * Pb_), 1.25 * Pb_)
                Pa, fa = Pb_, fb
                Pb_, fb = safe(Pn, Pa)
            if fb != fa and Pb_ != Pa:
                sat["dVdP"] = (fb - fa) / (Pb_ - Pa)
            return Pb_

        P = solve_P(P)
        rg, rl = cache[P]
        redo = False
        def unstable(r, root):
            x = r.phases[0].x
            if not m.stability(x, r.T, P)[0]:
                return True
            # the fast solve used EOS root `root`; the tangent-plane test cannot see the
            # same composition on the other root, so compare the two roots' residual Gibbs
            # energies g_res/RT = sum x ln(phi) (Michelsen & Mollerup 2007, ch. 2; study 15)
            xf = np.maximum(np.asarray(x, float), 1e-300)
            lnA, ZA = m._lnphi(xf, r.T, P, root)
            lnB, ZB = m._lnphi(xf, r.T, P, "V" if root == "L" else "L")
            return abs(ZA - ZB) > 1e-9 and float(xf @ lnB) < float(xf @ lnA) - 1e-9

        for key, r, root in (("G", rg, "V"), ("L", rl, "L")):
            if r is not None and r.kind == "fast1" and unstable(r, root):
                full[key] = True
                redo = True
        if redo:
            P = solve_P(P)
            rg, rl = cache[P]
        for key, r in (("G", rg), ("L", rl)):
            # stay in full-flash mode while the zone keeps splitting; retry fast mode
            # every 20 steps so a zone that has settled goes back to the cheap path
            sticky[key] = bool(r is not None and len(r.phases) > 1) or (full[key] and step % 20 != 0)
            last_split[key] = r if (r is not None and len(r.phases) > 1) else None
        G_H = G_H + VG0 * (P - P0s)
        L_H = L_H + VL0 * (P - P0s)

        # ---------------- phase transfer between zones
        # The zone that loses a phase keeps the other phase(s) exactly (same P, T and
        # phase composition), so its state is built directly; only a zone that
        # RECEIVES material needs a new (single-phase, fast) solve.
        from thermo_pr import FlashResult

        def kept_result(r, keep):
            ph = [p.copy() for p in r.phases if p.name in keep]
            bt = sum(p.beta for p in ph)
            for p in ph:
                p.beta /= bt
            x = sum(p.beta * np.asarray(p.x) for p in ph)
            return FlashResult(r.T, P, x, ph, kind="kept", names=m.names)

        rG, rL = rg, rl
        recvG = recvL = False
        if rg is not None and len(rg.phases) > 1:
            (kn, kH, _), (mn, mH, _) = split_phases(rg, G_n.sum(), keep=("vapour",))
            kH += G_H - (kH + mH)              # flash-tolerance residual stays in the zone
            if mn.sum() > 0 and kn.sum() > 1e-3:
                G_n, G_H, rG = kn, kH, kept_result(rg, ("vapour",))
                L_n, L_H, recvL = L_n + mn, L_H + mH, True
        if rl is not None and len(rl.phases) > 1 and rl.vapour is not None:
            (kn, kH, _), (mn, mH, _) = split_phases(rl, L_n.sum(), keep=("liquid", "aqueous"))
            kH += L_H - (kH + mH)
            if mn.sum() > 0 and kn.sum() > 1e-3:
                L_n, L_H, rL = kn, kH, kept_result(rl, ("liquid", "aqueous"))
                G_n, G_H, recvG = G_n + mn, G_H + mH, True
        if recvG:
            xg = G_n / G_n.sum()
            Tg0 = rG.T if rG is not None else (rl.T if rl is not None else Lz.T)
            rG = (fast_PH_single(m, xg, P, G_H / G_n.sum(), Tg0, "V") if rG is not None else None) or                 m.flash_PH(P, G_H / G_n.sum(), xg, T_guess=Tg0, init=rG)
        G = Zone("gas", G_n, G_H, rG) if (rG is not None and G_n.sum() > 1e-3) else None
        if L_n.sum() > 1e-3:
            if recvL or rL is None:
                xl, hl_ = L_n / L_n.sum(), L_H / L_n.sum()
                if rL is not None and len(rL.phases) == 1 and rL.phases[0].name == "liquid":
                    rL = fast_PH_single(m, xl, P, hl_, rL.T, "L") or                         m.flash_PH(P, hl_, xl, T_guess=rL.T, init=rL)
                else:
                    rL = m.flash_PH(P, hl_, xl, T_guess=Lz.T, init=rL)
            Lz = Zone("liq", L_n, L_H, rL)
        else:
            Lz = Zone("liq", np.zeros(len(names)), 0.0, G.r if G is not None else rG)
        if G is None:                       # gas zone vented / compressed away
            G = Zone("gas", np.zeros(len(names)), 0.0, Lz.r)

        # ---------------- gas zone vented away / compressed out: one dense zone remains
        if not G.empty and not Lz.empty and (G.mass < max(1.0, 5e-3 * m_gas0)
                                             or Lz.V > (1.0 - 1e-3) * V):
            L_n2, L_H2 = Lz.n + G.n, Lz.H + G.H
            rL2 = m.flash_PH(P, L_H2 / L_n2.sum(), L_n2 / L_n2.sum(), T_guess=Lz.T, init=None)
            Lz = Zone("liq", L_n2, L_H2, rL2)
            G = Zone("gas", np.zeros(len(names)), 0.0, rL2)
            last_split["G"], sticky["G"] = None, False

        # ---------------- boil-dry: a vanishing liquid zone joins the gas zone
        if not Lz.empty and (Lz.mass < max(1.0, 5e-3 * m_liq0) or geom.level(Lz.V) < 5e-3):
            G_n, G_H = G.n + Lz.n, G.H + Lz.H
            rG = m.flash_PH(P, G_H / G_n.sum(), G_n / G_n.sum(), T_guess=G.T, init=None)
            G = Zone("gas", G_n, G_H, rG)
            Lz = Zone("liq", np.zeros(len(names)), 0.0, G.r)
            last_split["L"], sticky["L"] = None, False

        # ---------------- wall areas follow the level (energy-conserving re-weighting)
        f_new = geom.wetted_perimeter_fraction(lvl()) if not Lz.empty else 0.0
        df = f_new - frac["wet"]
        if abs(df) > 1e-9:
            src, dst = ("dry", "wet") if df > 0 else ("wet", "dry")
            a = abs(df)
            if frac[dst] + a > 0:
                # energy-conserving mixing node by node: e(T) = integral of cp dT
                e_mix = (e_of_T(cols[dst].T) * frac[dst] + e_of_T(cols[src].T) * a) / (frac[dst] + a)
                cols[dst].T = np.interp(e_mix, E_TAB, T_TAB)
            frac[dst] += a
            frac[src] = max(frac[src] - a, 0.0)

        # sanity: stop at the first non-finite state with diagnostics
        if not (np.all(np.isfinite(cols["dry"].T)) and np.all(np.isfinite(cols["wet"].T))
                and np.isfinite(G.T) and np.isfinite(P) and np.isfinite(G.H)):
            raise FloatingPointError(
                f"non-finite state at t={time}: P={P}, G.T={G.T}, G.H={G.H}, G phases="
                f"{G.r.phase_names}, L phases={Lz.r.phase_names}, h_dry={info['h_dry']}, "
                f"Q_wg={info['Q_wg']}, dryT={cols['dry'].T[[0, -1]]}, wetT={cols['wet'].T[[0, -1]]}, "
                f"frac={frac}, yG={np.round(yG, 4)}")

        if time >= next_out - 1e-9:
            record(time)
            next_out += out_every

    meta = dict(fluid=" ".join(f"{k} {v:g}" for k, v in fl.items()) + " [PR two-phase]",
                V=V, m0=rows[0]["m_gas"] + rows[0]["m_liq"], rupture=rupture, opt=opt,
                names=names)
    return pd.DataFrame(rows), meta
