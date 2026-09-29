"""
vessel.py - transient simulation of a gas-filled vessel with fire heat input,
blowdown valve (BDV) and pressure safety valve (PSV).

Physics (published / textbook, no VessFire internals):
  Gas     : well-mixed, single phase, real-gas multi-component EOS (fluid.py).
            dm/dt = -mdot_out
            dU/dt = Q_wall->gas - mdot_out * h_gas        (open-system energy balance)
  Wall    : 1D radial conduction per wall column (heat_transfer.py), background
            and peak-zone columns weighted by area.
  Outer   : fire (FireBC), prescribed absorbed flux, or ambient exchange.
  Inner   : natural convection, correlation selectable (default Churchill-Chu).
  Valves  : orifice flow, real-gas homogeneous isentropic nozzle (default) or
            API 520 equations, times discharge coefficient.
  Stress  : membrane stresses vs strength * F(T_mean) at the hottest column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

from fluid import Fluid
from heat_transfer import (SIGMA, AmbientBC, FireBC, GuidelineFire, Material,
                           PrescribedFluxBC, WallColumn, VESSFIRE_LOG_NODES_105MM)

G_ACC = 9.81
P_ATM = 101325.0


# ============================================================
# OPTIONS
# ============================================================

@dataclass
class Options:
    flux: str = "prescribed"        # prescribed | blackbody | balance | none
    prescribed: dict | None = None  # {"t":..., "q":..., "q_rad":...} (W/m2)
    eps_surf_fire: float = 0.7      # steel emissivity for the fire model
    h_fire: float = 25.0            # fire-side convection coefficient
    eps_flame: float = 1.0
    h_corr: str = "churchill_chu"   # inner-wall correlation, see H_CORRELATIONS
    eos: str = "HEOS"               # HEOS (reference) | PR | SRK
    nozzle: str = "api520"          # api520 (API 520 gas equations, real Z and k) | hem
    dt: float | None = None         # default: Admin #Max_Timestep
    out_every: float | None = None  # default: Admin #Output_Frequence
    t_end: float | None = None      # default: Admin #Maxtime
    x_nodes: np.ndarray = field(default_factory=lambda: VESSFIRE_LOG_NODES_105MM)


# ============================================================
# HEAT TRANSFER CORRELATIONS (inner wall -> gas)
# ============================================================

H_CORRELATIONS = {
    "mcadams": (lambda Ra, Pr: 0.13 * Ra**(1/3) if Ra > 1e9 else 0.59 * Ra**0.25),
    "churchill_chu": (lambda Ra, Pr: (0.60 + 0.387 * Ra**(1/6)
                                      / (1 + (0.559 / Pr)**(9/16))**(8/27))**2),
    "woodfield_h2": (lambda Ra, Pr: 0.104 * Ra**0.352),
    "laminar": (lambda Ra, Pr: 0.59 * Ra**0.25),
    # Evans & Stefany (1966), Chem. Eng. Prog. Symp. Ser. 62(64):209 - transient
    # natural-convection heating of fluid inside closed containers, L = fluid height (= D)
    "evans_stefany": (lambda Ra, Pr: 0.55 * Ra**0.25),
}


def h_natural(fluid: Fluid, T_wall, T_gas, P, L, corr):
    dT = abs(T_wall - T_gas)
    if dT < 1e-3:
        return 0.0
    p = fluid.film_props(0.5 * (T_wall + T_gas), P)
    Pr = p["cp"] * p["mu"] / p["k"]
    Ra = G_ACC * abs(p["beta"]) * dT * L**3 * p["rho"]**2 * p["cp"] / (p["mu"] * p["k"])
    return H_CORRELATIONS[corr](Ra, Pr) * p["k"] / L


_AIR = CP.AbstractState("HEOS", "Air")


def h_ambient(T_s, T_env, D_out, spec):
    """Outside convection to still/moving air. spec > 0: fixed h; spec < 0: |spec|
    is a wind speed [m/s] - forced (Churchill-Bernstein) and natural (Churchill-Chu)
    convection combined as (h_f^3 + h_n^3)^(1/3)."""
    if spec > 0:
        return spec
    Tf = 0.5 * (T_s + T_env)
    _AIR.update(CP.PT_INPUTS, P_ATM, Tf)
    k, mu, rho, cp = _AIR.conductivity(), _AIR.viscosity(), _AIR.rhomass(), _AIR.cpmass()
    Pr = cp * mu / k
    Ra = G_ACC / Tf * abs(T_s - T_env) * D_out**3 * rho**2 * cp / (mu * k)
    Nu_n = (0.60 + 0.387 * Ra**(1/6) / (1 + (0.559 / Pr)**(9/16))**(8/27))**2
    v = abs(spec) if abs(spec) < 50 else 0.0
    Re = rho * v * D_out / mu
    Nu_f = 0.3 + 0.62 * Re**0.5 * Pr**(1/3) / (1 + (0.4 / Pr)**(2/3))**0.25 \
        * (1 + (Re / 282000)**(5/8))**0.8 if Re > 0 else 0.0
    return (Nu_n**3 + Nu_f**3)**(1/3) * k / D_out


class AmbientAuto(AmbientBC):
    """AmbientBC whose convection coefficient follows the surface temperature."""

    def __init__(self, T_env, eps, D_out, spec):
        super().__init__(T_env, eps, 5.0)
        self.D_out, self.spec = D_out, spec

    def __call__(self, T_s, t):
        self.h = h_ambient(T_s, self.T_env, self.D_out, self.spec)
        return super().__call__(T_s, t)


# ============================================================
# VALVES
# ============================================================

class Orifice:
    def __init__(self, fluid: Fluid, d: float, cd: float, model: str):
        self.fluid, self.A, self.cd, self.model = fluid, math.pi / 4 * d**2, cd, model

    def mdot(self, P, T, P_back, frac=1.0):
        if frac <= 0 or P <= P_back:
            return 0.0
        if self.model == "api520":
            G = self.fluid.api520_mass_flux(P, T, P_back)
        else:
            G, _ = self.fluid.isentropic_mass_flux(P, T, P_back)
        return frac * self.cd * self.A * G


class PSV:
    """Pressure safety valve with the opening characteristics of the VessFire manual
    (section 5.3.1 figure). Type index follows the manual's listing order
    "trapezoidal, triangular or square":
      0 trapezoidal : rising - closed until Pset, then linear 0 -> 100 % at Pfull;
                      falling - stays at its opening until Preseat, then closes.
      1 triangular  : rising - closed until Pset, then 100 %;
                      falling - linear from 100 % at Pset to 0 % at Preseat.
      2 square      : rising - closed until Pset, then 100 %;
                      falling - stays 100 % until Preseat, then closes."""

    def __init__(self, fluid, spec, model):
        self.orif = Orifice(fluid, spec["d"], spec["cd"], model)
        self.s = spec
        self.open_frac = 0.0

    def frac(self, P):
        s, f = self.s, self.open_frac
        Ps, Pr, Pf = s["p_set"], s["p_reseat"], s["p_full"]
        if f <= 0.0:                                   # closed: opens only at Pset
            if P >= Ps:
                f = min(max((P - Ps) / max(Pf - Ps, 1.0), 1e-6), 1.0) if s["type"] == 0 else 1.0
        elif s["type"] == 0:                           # trapezoidal
            f = 0.0 if P < Pr else max(f, min((P - Ps) / max(Pf - Ps, 1.0), 1.0))
        elif s["type"] == 1:                           # triangular
            f = 1.0 if P >= Ps else max((P - Pr) / max(Ps - Pr, 1.0), 0.0)
        else:                                          # square
            f = 0.0 if P < Pr else 1.0
        self.open_frac = f
        return f


# ============================================================
# STRESS
# ============================================================

def stresses(P, D, t, ext_long_mpa):
    Pg = (P - P_ATM) / 1e6
    rm = 0.5 * D + 0.5 * t
    s_h = Pg * rm / t
    s_l = Pg * rm / (2 * t) + ext_long_mpa
    s_r = -0.5 * Pg
    vm = math.sqrt(0.5 * ((s_h - s_l)**2 + (s_l - s_r)**2 + (s_r - s_h)**2))
    tr = max(s_h, s_l, s_r) - min(s_h, s_l, s_r)
    return s_h, s_l, vm, tr


# ============================================================
# SIMULATION
# ============================================================

def simulate(case: dict, opt: Options) -> tuple[pd.DataFrame, dict]:
    s, hl, adm = case["seg"], case["hl"], case["admin"]
    fluid = Fluid(s["fluid"], opt.eos)
    mat = Material.from_vessfire_db(s["material"])
    D, t_w, L = s["D"], s["t"], s["L"]
    D_out = D + 2 * t_w
    V = math.pi / 4 * D**2 * L
    A_in, A_out = math.pi * D * L, math.pi * D_out * L

    ambient = AmbientAuto(s["T_env"], s["eps_surf"], D_out, s["h_out"])
    f_peak = (hl.get("xi_end", 0) - hl.get("xi_start", 0)) * hl.get("circ_deg", 0) / 360.0
    ser = hl["series"]

    def outer_bc(which):
        if opt.flux == "none" or not np.any(ser[:, 1:] > 0):
            return ambient
        if opt.flux == "prescribed":
            p = opt.prescribed
            return PrescribedFluxBC(p["t"], p["q"], p.get("q_rad"))
        fire = GuidelineFire(eps_flame=opt.eps_flame, eps_surf=opt.eps_surf_fire,
                             h_flame=opt.h_fire, T_ref=s["T_shell"],
                             t_air_mode="blackbody" if opt.flux == "blackbody" else "balance")
        col = 2 if which == "peak" else 1
        return FireBC(fire, ser[:, 0], ser[:, col] * 1e3, after=ambient)

    cols = {"background": dict(frac=1.0 - f_peak), "peak": dict(frac=f_peak)}
    cols = {k: v for k, v in cols.items() if v["frac"] > 0}
    for k, v in cols.items():
        v["col"] = WallColumn(mat, D / 2, opt.x_nodes, outer_bc(k), s["T_shell"])

    P, T = s["P0"], s["T0"]
    rho = fluid.rho_PT(P, T)
    m = rho * V
    U = m * fluid.at_P_T(P, T).umass()
    st = fluid.state_rho_u(rho, U / m, T)

    bdv = Orifice(fluid, s["bdv"]["d"], s["bdv"]["cd"], opt.nozzle) if s.get("bdv") else None
    psv = PSV(fluid, s["psv"], opt.nozzle) if s.get("psv") else None
    Pb = s["back_pressure"]

    dt = opt.dt or float(adm.get("max_timestep", 1.0))
    t_end = opt.t_end or float(adm["maxtime"])
    out_every = opt.out_every or float(adm.get("output_frequence", 10))
    n_steps = int(round(t_end / dt))

    E_wall0 = sum(v["col"].energy() * v["frac"] for v in cols.values()) * L
    m0, U0 = m, U
    Q_out_in = 0.0        # heat through outer surface
    H_out = 0.0           # enthalpy leaving through valves
    released = 0.0
    rows, rupture = [], {"vonMises": None, "Tresca": None}
    h_in = {k: 0.0 for k in cols}
    md_bdv = md_psv = 0.0

    def record(time):
        fluid.check_single_phase(st["P"], st["T"])
        hot = max(cols, key=lambda k: cols[k]["col"].T_mean())
        Tm = cols[hot]["col"].T_mean()
        f = mat.f_uts_at(Tm) if s.get("stress_type", "U") == "U" else mat.f_yield_at(Tm)
        allow = s["strength_mpa"] * s.get("stress_factor", 1.0) * f
        s_h, s_l, vm, tr = stresses(st["P"], D, t_w, s.get("ext_long_mpa", 0.0))
        for crit, val in (("vonMises", vm), ("Tresca", tr)):
            if rupture[crit] is None and val >= allow:
                rupture[crit] = time
        E_wall = sum(v["col"].energy() * v["frac"] for v in cols.values()) * L
        bal = (E_wall - E_wall0) + (U - U0) + H_out - Q_out_in
        r = dict(Time=time, P_bara=st["P"] / 1e5, P_barg=(st["P"] - P_ATM) / 1e5,
                 T_gas_C=st["T"] - 273.15, m_gas=m, rho_gas=m / V, Z=st["Z"],
                 mdot_bdv=md_bdv, mdot_psv=md_psv, released=released,
                 sigma_vM=vm, sigma_Tresca=tr, sigma_allow=allow, T_mean_hot_C=Tm - 273.15,
                 Q_gas_kW=sum(v["col"].q_in * A_in * v["frac"] for v in cols.values()) / 1e3,
                 energy_err_pct=100 * bal / max(abs(Q_out_in) + H_out, 1.0))
        for k, v in cols.items():
            c = v["col"]
            r[f"{k}_T_out_C"] = c.T_outer - 273.15
            r[f"{k}_T_in_C"] = c.T_inner - 273.15
            r[f"{k}_T_mean_C"] = c.T_mean() - 273.15
            r[f"{k}_q_net_kW"] = c.q_net_out / 1e3
            r[f"{k}_q_rad_kW"] = c.q_rad_out / 1e3
            r[f"{k}_q_conv_kW"] = c.q_conv_out / 1e3
            r[f"{k}_h_in"] = h_in[k]
            for i, Ti in enumerate(c.T):
                r[f"{k}_T{i+1}_C"] = Ti - 273.15
        rows.append(r)

    for v in cols.values():                               # t=0 boundary flux for output
        c = v["col"]
        c.q_net_out, _, c.q_rad_out, c.q_conv_out = c.outer(c.T_outer, 0.0)
    if bdv and s["bdv"]["delay"] <= 0:                    # t=0 valve flow for output
        md_bdv = bdv.mdot(st["P"], st["T"], Pb)
    record(0.0)
    next_out = out_every

    for n in range(1, n_steps + 1):
        time = n * dt
        t_mid = time - 0.5 * dt
        P, T = st["P"], st["T"]

        # valves (flow evaluated at start-of-step state)
        md_bdv = bdv.mdot(P, T, Pb) if bdv and t_mid >= s["bdv"]["delay"] else 0.0
        md_psv = psv.orif.mdot(P, T, Pb, psv.frac(P)) if psv else 0.0
        mdot = md_bdv + md_psv

        # wall
        Q = 0.0
        for k, v in cols.items():
            c = v["col"]
            h_in[k] = h_natural(fluid, c.T_inner, T, P, D, opt.h_corr)
            q_in = c.step(dt, t_mid, T, h_in[k])
            Q += q_in * A_in * v["frac"]
            Q_out_in += c.q_net_out * A_out * v["frac"] * dt

        # gas
        dm = mdot * dt
        H_out += dm * st["h"]
        released += dm
        m -= dm
        U += (Q - mdot * st["h"]) * dt
        st = fluid.state_rho_u(m / V, U / m, T)

        if time >= next_out - 1e-9:
            record(time)
            next_out += out_every

    meta = dict(fluid=fluid.label, V=V, m0=m0, f_peak=f_peak, rupture=rupture, opt=opt,
                A_in=A_in, A_out=A_out)
    return pd.DataFrame(rows), meta


def prescribed_from_vessfire(vf: pd.DataFrame) -> dict:
    """VessFire's absorbed-flux history (Energy, RadIn_max columns) as an input."""
    return {"t": vf.Time.values - 0.5, "q": vf.Energy.values, "q_rad": vf.RadIn_max.values}
