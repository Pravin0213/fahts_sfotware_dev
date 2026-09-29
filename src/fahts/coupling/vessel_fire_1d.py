"""Vessel in fire: two-zone non-equilibrium contents coupled to 1-D wall columns.

Formulation (published blowdown practice: Haque et al. 1992; Mahgerefteh & Wong 1999;
Speranza & Terenzi 2005):

Zones
  gas zone  G : moles n_G [mol], enthalpy H_G [J]        (vapour, may form fog)
  liquid    L : moles n_L [mol], enthalpy H_L [J]        (HC liquid and/or free water)
  common pressure P;  V_G + V_L = V_vessel.
Each zone is internally in equilibrium (Peng-Robinson flash, ``fahts.thermo``); the two
zones exchange heat and mass only through the interface and phase transfer:
  - rain-out: liquid/aqueous phases formed in G are moved to L
  - boil-off / flashing: vapour formed in L is moved to G
  - interface evaporation/condensation from the interface energy balance
    Q_li = Q_ig + mdot * h_fg   (liquid side convection, gas side convection)

Wall: two radial 1-D columns (``fahts.wall.column_1d``), "dry" (in contact with the gas)
and "wet" (in contact with the liquid), weighted by the wetted perimeter fraction. Outer
surface: fire (``fahts.fire``) or ambient. Relief: blowdown valve + line and PSV
(``fahts.relief``).

Per time step dt (``VesselFireModel.step``; rates explicit from the start-of-step state,
wall implicit):
  1. saturation properties of the liquid pool (refreshed every ``sat_every`` s)
  2. valve flows from the gas zone (valves at the top)
  3. dry wall -> gas (natural convection + internal radiation), wall column step
  4. wet wall -> liquid (free convection / nucleate / film boiling, linearised), step
  5. interface exchange
  6. zone enthalpy updates: H_z* = H_z + (Q_z - mdot_out,z h_z) dt (+ interface mass)
  7. pressure: find P with  sum_z N_z v_z(P, h_z(P)) = V,
     h_z(P) = (H_z* + V_z,old (P - P_old)) / N_z   (dH = dQ + V dP for each zone,
     which makes total energy change = Q - H_out exactly since sum V_z = V)
  8. phase transfer between zones at the new P (volume and energy move with the phase)
  9. merge a vanishing zone (vented / compressed-out gas, boiled-dry liquid)
 10. level -> wetted fraction -> re-weight dry/wet wall columns (energy conserving)

Known issues (kept for behaviour-preserving port): docs/process_model_known_issues.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fahts.coupling.options import VesselFireOptions
from fahts.fire import AmbientAuto, FireBC, GuidelineFire
from fahts.materials import SteelTable
from fahts.process.fluid_properties import film_gas, phase_dict, sat_props
from fahts.process.geometry import VesselGeometry
from fahts.process.inner_ht.boiling_curve import (boiling_flux_and_derivative, linearised_bc,
                                                  with_derivative)
from fahts.process.inner_ht.interface import interface_exchange
from fahts.process.inner_ht.internal_radiation import rad_network
from fahts.process.inner_ht.natural_convection import H_CORRELATIONS, h_free
from fahts.process.inner_ht.nucleate_only import nucleate_only_flux
from fahts.process.zones import Zone, fast_PH_multi, fast_PH_single, split_phases
from fahts.relief import blowdown
from fahts.relief.ideal_gas import mdot_orifice_line, orifice_G
from fahts.relief.psv import PSVOpening
from fahts.rupture import membrane_stresses
from fahts.thermo import FlashResult, PRMixture
from fahts.wall.column_1d import REFERENCE_NODES_105MM, WallColumn

G_ACC = 9.81          # see docs/process_model_known_issues.md #5 (two gravity values)
P_ATM = 101325.0


@dataclass
class StepContext:
    """Values computed during one time step and handed from one stage to the next."""
    time: float
    t_mid: float
    gv: object = None                 # vapour phase the valves draw from
    yG: np.ndarray | None = None      # its composition
    src_is_gas: bool = True           # valves draw from the gas zone (else dense liquid)
    T_src: float = 0.0
    liq_present: bool = False
    sd: dict | None = None            # saturation properties of the pool (None: no boiling)
    Mv: float = 0.0
    mdot: float = 0.0
    h_dry: float = 0.0
    Q_wg: float = 0.0                 # dry wall -> gas [W]
    Q_rad_l: float = 0.0              # radiation dry wall -> liquid surface [W]
    Q_wl: float = 0.0                 # wet wall -> liquid [W]
    Q_ig: float = 0.0                 # interface -> gas [W]
    Q_li: float = 0.0                 # liquid -> interface [W]
    m_ev: float = 0.0
    n_ev: np.ndarray | None = None
    H_evL: float = 0.0
    H_evG: float = 0.0
    G_n: np.ndarray | None = None     # zone contents after flows, before the pressure update
    G_H: float = 0.0
    L_n: np.ndarray | None = None
    L_H: float = 0.0
    VG0: float = 0.0
    VL0: float = 0.0
    P0s: float = 0.0
    full: dict = field(default_factory=dict)
    prevL: object = None
    cache: dict = field(default_factory=dict)
    rg: object = None                 # zone flash results at the new pressure
    rl: object = None


class VesselFireModel:
    """Transient vessel-in-fire simulation for one case (see module docstring).

    case : parsed input deck (``seg``, ``hl``, ``admin`` dicts; see ``fahts.io``)
    opt  : model options
    mat  : steel property table of the shell
    """

    def __init__(self, case: dict, opt: VesselFireOptions, mat: SteelTable):
        s, hl, adm = case["seg"], case["hl"], case["admin"]
        self.case, self.opt, self.mat = case, opt, mat
        self.s, self.hl, self.adm = s, hl, adm
        fl = dict(s["fluid"])
        self.fl = fl
        pseudo = {k: dict(sg=v["rd"], Tb=v["Tb"]) for k, v in s.get("pseudo", {}).items()}
        m = PRMixture(fl, pseudo=pseudo or None)
        self.m = m
        names = m.names
        self.names = names
        z0 = np.array([fl[n] if n in fl else fl.get(n.upper(), 0.0) for n in names], float)
        z0 /= z0.sum()

        D, t_w, L = s["D"], s["t"], s["L"]
        self.D, self.t_w, self.L = D, t_w, L
        self.D_out = D + 2 * t_w
        geom = VesselGeometry(D=D, L=L, orientation="horizontal", head="flat")
        self.geom = geom
        V = geom.V_total
        self.V = V
        self.A_in, self.A_out = math.pi * D * L, math.pi * self.D_out * L
        self.ser = hl["series"]
        self.fire_on = opt.flux != "none" and np.any(self.ser[:, 1:] > 0)

        # ------------------------------------------------------------ initial state
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
            nL += (V_oil / Lp.v) * np.asarray(Lp.x)
            HL += (V_oil / Lp.v) * Lp.h
        if Wp is not None and V_w > 0:
            nL += (V_w / Wp.v) * np.asarray(Wp.x)
            HL += (V_w / Wp.v) * Wp.h
        if Vp is None:
            raise ValueError("initial state has no vapour phase - liquid-full vessels not supported")
        self.P = P
        self.G = Zone("gas", nG, HG, m.flash_PH(P, HG / nG.sum(), nG / nG.sum(), T_guess=T0))
        self.Lz = Zone("liq", nL, HL, m.flash_PH(P, HL / nL.sum(), nL / nL.sum(), T_guess=T0)) \
            if nL.sum() > 1e-3 else Zone("liq", nL, 0.0, r0)

        # ------------------------------------------------------------ wall
        f_wet = geom.wetted_perimeter_fraction(self._level()) if not self.Lz.empty else 0.0
        self.cols = {
            "dry": WallColumn(mat, D / 2, REFERENCE_NODES_105MM, self._outer_bc(), s["T_shell"]),
            "wet": WallColumn(mat, D / 2, REFERENCE_NODES_105MM, self._outer_bc(), s["T_shell"])}
        self.frac = {"dry": 1.0 - f_wet, "wet": f_wet}
        self.T_TAB = np.linspace(50.0, 2000.0, 7801)
        self.E_TAB = np.concatenate(([0.0], np.cumsum(
            0.5 * (mat.cp_at(self.T_TAB[1:]) + mat.cp_at(self.T_TAB[:-1])) * np.diff(self.T_TAB))))

        # ------------------------------------------------------------ valves
        self.Pb = s["back_pressure"]
        self.bdv = s.get("bdv")
        self.psv = PSVOpening(s["psv"]) if s.get("psv") else None
        self.A_psv = math.pi / 4 * s["psv"]["d"] ** 2 if s.get("psv") else 0.0

        # ------------------------------------------------------------ time and bookkeeping
        self.dt = opt.dt or float(adm.get("max_timestep", 1.0))
        t_end = opt.t_end or float(adm["maxtime"])
        self.out_every = opt.out_every or float(adm.get("output_frequence", 10))
        self.n_steps = int(round(t_end / self.dt))

        self.E_wall0 = sum(c.energy() * self.frac[k] for k, c in self.cols.items()) * L
        self.m_liq0 = self.Lz.mass if not self.Lz.empty else 0.0
        self.m_gas0 = self.G.mass
        self.U0 = (self.G.H - P * self.G.V) + \
            ((self.Lz.H - P * self.Lz.V) if not self.Lz.empty else 0.0)
        self.Q_fire = self.H_out = 0.0
        self.Q_sink = 0.0            # heat removed by a free-water sink pool [J]
        self.throughput = 0.0        # heat + enthalpy throughput, scale for the energy error
        self.sticky = {"G": False, "L": False}
        self.last_split = {"G": None, "L": None}
        self.released = 0.0
        self.rows = []
        self.rupture = {"vonMises": None, "Tresca": None}
        self.sat = {"t": -1e9, "d": None}
        self.info = dict(h_dry=0.0, h_wet=0.0, regime="", Q_wg=0.0, Q_wl=0.0, Q_ig=0.0,
                         Q_li=0.0, m_evap=0.0, md_bdv=0.0, md_psv=0.0)
        self.next_out = 0.0

    # ================================================================ run
    def run(self) -> tuple[pd.DataFrame, dict]:
        for c in self.cols.values():
            c.q_net_out, _, c.q_rad_out, c.q_conv_out = c.outer(c.T_outer, 0.0)
        self._record(0.0)
        self.next_out = self.out_every
        for step in range(1, self.n_steps + 1):
            self.step(step)
        meta = dict(fluid=" ".join(f"{k} {v:g}" for k, v in self.fl.items()) + " [PR two-phase]",
                    V=self.V, m0=self.rows[0]["m_gas"] + self.rows[0]["m_liq"],
                    rupture=self.rupture, opt=self.opt, names=self.names)
        return pd.DataFrame(self.rows), meta

    def step(self, step: int) -> None:
        """Advance one time step (stages 1-10 of the module docstring)."""
        time = step * self.dt
        c = StepContext(time=time, t_mid=time - 0.5 * self.dt)
        c.gv = self._gas_vapour()
        c.yG = np.asarray(c.gv.x)
        c.src_is_gas = not self.G.empty
        c.T_src = self.G.T if c.src_is_gas else self.Lz.T
        c.liq_present = not self.Lz.empty

        self._refresh_saturation(c)
        self._valve_flows(c)
        self._dry_wall(c)
        self._wet_wall(c)
        self._interface(c)
        self._zone_balances(c)
        self._solve_pressure(c, step)
        self._phase_transfer(c)
        self._merge_vanishing_zones()
        self._reweight_wall()
        self._check_finite(c)

        if time >= self.next_out - 1e-9:
            self._record(time)
            self.next_out += self.out_every

    # ================================================================ helpers
    def _outer_bc(self):
        s, opt = self.s, self.opt
        amb = AmbientAuto(s["T_env"], s["eps_surf"], self.D_out, s["h_out"])
        if not self.fire_on:
            return amb
        fire = GuidelineFire(eps_flame=1.0, eps_surf=opt.eps_surf_fire, h_flame=opt.h_fire,
                             T_ref=s["T_shell"],
                             t_air_mode="blackbody" if opt.flux == "blackbody" else "balance")
        return FireBC(fire, self.ser[:, 0], self.ser[:, 1] * 1e3, after=amb)

    def _level(self) -> float:
        return self.geom.level(self.Lz.V) if not self.Lz.empty else 0.0

    def _e_of_T(self, T):
        return np.interp(T, self.T_TAB, self.E_TAB)

    def _gas_vapour(self):
        G, Lz = self.G, self.Lz
        if G.empty and not Lz.empty:
            return Lz.r.phases[0]
        return G.r.vapour or G.r.phases[0]

    # ================================================================ output
    def _record(self, time: float) -> None:
        s, mat, cols, frac, info, P = self.s, self.mat, self.cols, self.frac, self.info, self.P
        G, Lz = self.G, self.Lz
        hot = max(cols, key=lambda k: cols[k].T_mean() if frac[k] > 0 else -1)
        Tm = cols[hot].T_mean()
        f = mat.f_uts_at(Tm) if s.get("stress_type", "U") == "U" else mat.f_yield_at(Tm)
        allow = s["strength_mpa"] * s.get("stress_factor", 1.0) * f
        s_h, s_l, vm, tr = membrane_stresses(P, self.D, self.t_w, s.get("ext_long_mpa", 0.0))
        for crit, val in (("vonMises", vm), ("Tresca", tr)):
            if self.rupture[crit] is None and val >= allow:
                self.rupture[crit] = time
        E_wall = sum(c.energy() * frac[k] for k, c in cols.items()) * self.L
        U = (G.H - P * G.V) + ((Lz.H - P * Lz.V) if not Lz.empty else 0.0)
        bal = (E_wall - self.E_wall0) + (U - self.U0) + self.H_out + self.Q_sink - self.Q_fire
        gv = self._gas_vapour()
        r = dict(Time=time, P_bara=P / 1e5, P_barg=(P - P_ATM) / 1e5,
                 T_gas_C=G.T - 273.15, T_liq_C=(Lz.T - 273.15) if not Lz.empty else np.nan,
                 m_gas=G.mass, m_gas_zone=G.mass, m_liq=Lz.mass, level=self._level(),
                 f_wet=frac["wet"], MW_gas=gv.M * 1e3, mdot_bdv=info["md_bdv"],
                 mdot_psv=info["md_psv"], released=self.released, sigma_vM=vm,
                 sigma_Tresca=tr, sigma_allow=allow, T_mean_hot_C=Tm - 273.15,
                 h_dry=info["h_dry"], h_wet=info["h_wet"], regime=info["regime"],
                 Q_wg_kW=info["Q_wg"] / 1e3, Q_wl_kW=info["Q_wl"] / 1e3,
                 Q_ig_kW=info["Q_ig"] / 1e3, Q_li_kW=info["Q_li"] / 1e3, m_evap=info["m_evap"],
                 Q_rad_liq_kW=info.get("Q_rad_l", 0.0) / 1e3,
                 Q_rad_gas_kW=info.get("Q_rad_g", 0.0) / 1e3, h_rad=info.get("h_rad", 0.0),
                 energy_err_MJ=bal / 1e6, Q_fire_cum_MJ=self.Q_fire / 1e6,
                 Q_sink_MJ=self.Q_sink / 1e6,
                 energy_err_pct=100 * bal / max(self.throughput, 1.0))
        for k, c in cols.items():
            key = "background" if k == "dry" else "wet"
            r[f"{key}_T_out_C"] = c.T_outer - 273.15
            r[f"{key}_T_in_C"] = c.T_inner - 273.15
            r[f"{key}_T_mean_C"] = c.T_mean() - 273.15
            r[f"{key}_q_net_kW"] = c.q_net_out / 1e3
            for i, Ti in enumerate(c.T):
                r[f"{key}_T{i+1}_C"] = Ti - 273.15
        self.rows.append(r)

    # ================================================================ 1. saturation
    def _refresh_saturation(self, c: StepContext) -> None:
        sat, Lz, m, P = self.sat, self.Lz, self.m, self.P
        if c.liq_present and c.time - sat["t"] >= self.opt.sat_every:
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
                except Exception:  # noqa: BLE001 - no bubble point at this P (supercritical)
                    sd_new = None
            if sd_new is not None:
                sat["Tsat"] = sd_new["T_sat"]
                # no genuine boiling when the pool is near/above its critical region
                # (latent heat collapsing): treat the wetted wall as single-phase
                if not water_pool and (sd_new["h_fg"] < 30e3 or sd_new["T_sat"] < Lz.T - 50.0):
                    sd_new = None
            sat["d"], sat["t"] = sd_new, c.time
            if sd_new is not None:
                sat["last"] = sd_new
        c.sd = sat["d"]

    # ================================================================ 2. valves
    def _valve_flows(self, c: StepContext) -> None:
        s, opt, P, Pb, bdv, psv = self.s, self.opt, self.P, self.Pb, self.bdv, self.psv
        gv = c.gv
        Z, kk, Mv = gv.Z, gv.cp / gv.cv, gv.M
        c.Mv = Mv
        if bdv and c.t_mid >= bdv["delay"]:
            line = s.get("bdv_line") if opt.use_line else None
            if opt.valve_model == "b1":
                md_bdv = blowdown.mdot(P, c.T_src, c.yG, Pb, bdv["cd"], bdv["d"], line,
                                       model=self.m, names=self.names,
                                       line_id="d" if opt.line_diameter == "inner" else "d-2t")
            else:
                if line and opt.line_diameter == "inner":
                    line = dict(line, t=0.0)
                md_bdv = mdot_orifice_line(P, self.G.T, Pb, Z, kk, Mv, bdv["cd"], bdv["d"], line,
                                           mu=self.sat.get("pg", {}).get("mu", 1.2e-5))
        else:
            md_bdv = 0.0
        if psv and not c.src_is_gas and opt.psv_liquid == "liquid":
            md_psv = self.A_psv * s["psv"]["cd"] * math.sqrt(2.0 * gv.rho * max(P - Pb, 0.0)) \
                * psv.frac(P)
        else:
            md_psv = self.A_psv * s["psv"]["cd"] * orifice_G(P, c.T_src, Pb, Z, kk, Mv) \
                * psv.frac(P) if psv else 0.0
        c.mdot = md_bdv + md_psv
        self.info["md_bdv"], self.info["md_psv"] = md_bdv, md_psv

    # ================================================================ 3. dry wall -> gas
    def _dry_wall(self, c: StepContext) -> None:
        opt, m, G, Lz, P, frac, info = self.opt, self.m, self.G, self.Lz, self.P, self.frac, \
            self.info
        D, dt = self.D, self.dt
        cd = self.cols["dry"]
        pg = film_gas(m, c.yG, 0.5 * (cd.T_inner + G.T), P)
        self.sat["pg"] = pg
        dTg = abs(cd.T_inner - G.T)
        if dTg > 1e-3:
            Pr = pg["cp"] * pg["mu"] / pg["k"]
            Ra = G_ACC * abs(pg["beta"]) * dTg * D**3 * pg["rho"]**2 * pg["cp"] / (pg["mu"] * pg["k"])
            h_dry = H_CORRELATIONS[opt.h_corr](Ra, Pr) * pg["k"] / D
        else:
            h_dry = 0.0
        T_fl = G.T if c.src_is_gas else Lz.T
        Q_rad_l = Q_rad_g = 0.0
        if opt.rad_internal and frac["dry"] > 0 and c.src_is_gas:
            A1 = self.A_in * frac["dry"]
            A3 = self.geom.interface_area(self.geom.level(Lz.V)) if not Lz.empty else 0.0
            T3 = Lz.T if not Lz.empty else G.T
            T1 = cd.T_inner
            Q1, Q3, Qg = rad_network(T1, T3, G.T, A1, A3, opt.eps_wall_in, opt.eps_liq,
                                     opt.eps_gas)
            Q1b = rad_network(T1 + 0.05, T3, G.T, A1, A3, opt.eps_wall_in, opt.eps_liq,
                              opt.eps_gas)[0]
            h_r = max((Q1b - Q1) / 0.05 / A1, 1e-6)             # W/m2K, radiative conductance
            T_ref_r = T1 - (Q1 / A1) / h_r
            h_tot = h_dry + h_r
            T_eff = (h_dry * T_fl + h_r * T_ref_r) / h_tot
            cd.step(dt, c.t_mid, T_eff, h_tot)
            Q1_new = h_r * (cd.T_inner - T_ref_r) * A1          # radiation actually leaving the wall
            # the liquid surface receives its network share (scaled to the implicit wall
            # value); the gas absorbs the remainder, so energy is conserved exactly
            scale = Q1_new / Q1 if abs(Q1) > 1.0 else 1.0
            Q_rad_l = -Q3 * scale
            Q_rad_g = Q1_new - Q_rad_l
            Q_wg = h_dry * (cd.T_inner - T_fl) * A1 + Q_rad_g
            info["h_rad"] = h_r
        else:
            q = cd.step(dt, c.t_mid, T_fl, h_dry)
            Q_wg = q * self.A_in * frac["dry"]
        self.Q_fire += cd.q_net_out * self.A_out * frac["dry"] * dt
        info["h_dry"] = h_dry
        info["Q_rad_l"], info["Q_rad_g"] = Q_rad_l, Q_rad_g
        c.h_dry, c.Q_wg, c.Q_rad_l = h_dry, Q_wg, Q_rad_l

    # ================================================================ 4. wet wall -> liquid
    def _wet_wall(self, c: StepContext) -> None:
        opt, m, G, Lz, P, frac, info, sat = self.opt, self.m, self.G, self.Lz, self.P, \
            self.frac, self.info, self.sat
        geom, dt, t_mid = self.geom, self.dt, c.t_mid
        cw = self.cols["wet"]
        Q_wl = 0.0
        sd = c.sd
        if (c.liq_present and sd is None and opt.wet_above_crit == "boiling"
                and sat.get("last") is not None
                and not (m.iw is not None and (Lz.n / Lz.N)[m.iw] > 0.99)):
            # VessFire-like: boiling-level heat transfer continues with the pool taken to be
            # at saturation, using the last valid saturation property set
            sd = dict(sat["last"], T_sat=Lz.T, P=P)
        if c.liq_present and frac["wet"] > 0 and sd is None:
            # single-phase natural convection to a dense (compressed / supercritical) pool
            Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
            # film-temperature properties (as on the gas side): for a dense / near-critical
            # pool the properties change strongly between bulk and wall temperature
            T_film = 0.5 * (cw.T_inner + Lz.T)
            try:
                pl = phase_dict(m, m.phase_props(Lph.x, T_film, P, "L")) \
                    if Lph.name != "aqueous" else phase_dict(m, Lph)
            except Exception:  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001
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
                        except Exception:  # noqa: BLE001
                            pass
                        Pr_l = cp_use * pl["mu"] / pl["k"]
                    Ra_l = G_ACC * drho * Lnc**3 * pl["rho"] * cp_use / (pl["mu"] * pl["k"])
                else:
                    Ra_l = G_ACC * abs(pl["beta"]) * dTl * Lnc**3 * pl["rho"]**2 * pl["cp"] \
                        / (pl["mu"] * pl["k"])
                h_w = H_CORRELATIONS[opt.h_corr_liq or opt.h_corr](Ra_l, Pr_l) * pl["k"] / Lnc
            else:
                h_w = 0.0
            cw.step(dt, t_mid, Lz.T, h_w)
            Q_wl = h_w * (cw.T_inner - Lz.T) * self.A_in * frac["wet"]
            info["h_wet"], info["regime"] = h_w, "single-phase"
        elif c.liq_present and frac["wet"] > 0 and sd is not None:
            Lph = Lz.r.liquid or Lz.r.aqueous or Lz.r.phases[0]
            pl = phase_dict(m, Lph)
            h_l = geom.level(Lz.V)
            Lnc = geom.liquid_char_length(h_l)
            try:
                if opt.wet_boiling == "nucleate_only":
                    q0, dq = with_derivative(
                        lambda Tw: nucleate_only_flux(Tw, Lz.T, sd, pl, Lnc), cw.T_inner)
                    reg = "nucleate_only"
                else:
                    q0, dq, reg = boiling_flux_and_derivative(cw.T_inner, Lz.T, sd, pl, Lnc)
            except Exception:  # noqa: BLE001
                q0, dq, reg = 0.0, 1.0, "fail"
            h_eff, T_ref = linearised_bc(q0, dq, cw.T_inner)
            cw.step(dt, t_mid, T_ref, h_eff)
            Q_wl = h_eff * (cw.T_inner - T_ref) * self.A_in * frac["wet"]
            info["h_wet"], info["regime"] = h_eff, reg
        else:
            cw.step(dt, t_mid, G.T, c.h_dry)               # keeps the (zero-area) column alive
        self.Q_fire += cw.q_net_out * self.A_out * frac["wet"] * dt
        c.sd, c.Q_wl = sd, Q_wl

    # ================================================================ 5. interface
    def _interface(self, c: StepContext) -> None:
        opt, m, G, Lz, P, dt, geom = self.opt, self.m, self.G, self.Lz, self.P, self.dt, \
            self.geom
        sd, yG = c.sd, c.yG
        Q_ig = Q_li = m_ev = 0.0
        n_ev = np.zeros(len(self.names))
        H_evL = H_evG = 0.0
        if opt.interface and c.liq_present and not G.empty and \
                (sd is not None or not opt.interface_mass):
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
                    hg = h_free(pgi, G.T - Ti, Ls, "hot_down" if G.T > Ti else "hot_up")
                    hl = h_free(pli, Lz.T - Ti, Ls, "hot_up" if Lz.T > Ti else "hot_down")
                    Ti_new = (hg * G.T + hl * Lz.T) / max(hg + hl, 1e-9)
                    if abs(Ti_new - Ti) < 1e-4:
                        break
                    Ti = Ti_new
                Qs = hl * Ai * (Lz.T - Ti)             # liquid -> interface = interface -> gas
                ie = dict(Q_ig=Qs, Q_li=Qs, mdot_evap=0.0)
            else:
                ie = interface_exchange(G.T, Lz.T, sd, film_gas(m, yG, G.T, P),
                                        phase_dict(m, Lph), Ai, Ls)
            Q_ig, Q_li, m_ev = ie["Q_ig"], ie["Q_li"], ie["mdot_evap"] * dt
            # limit phase change to what the zones can supply in a step
            if m_ev > 0:
                m_ev = min(m_ev, 0.2 * Lz.mass)
                ycomp = np.asarray(sd.get("y", yG))
                Mev = float(ycomp @ m.Mw)
                n_ev = m_ev / Mev * ycomp                 # evaporating (incipient vapour) moles
                n_ev = np.minimum(n_ev, 0.5 * Lz.n)
                H_evL, H_evG = m_ev * sd["h_l"], m_ev * sd["h_v"]
            elif m_ev < 0:
                m_ev = max(m_ev, -0.2 * G.mass)
                n_ev = m_ev / (yG @ m.Mw) * yG             # condensing gas moles (negative)
                H_evL, H_evG = m_ev * sd["h_l"], m_ev * sd["h_v"]
        self.info.update(Q_wg=c.Q_wg, Q_wl=c.Q_wl, Q_ig=Q_ig, Q_li=Q_li, m_evap=m_ev / dt)
        self.throughput += (abs(c.Q_wg) + abs(c.Q_wl) + abs(Q_ig) + abs(Q_li)
                            + abs(c.Q_rad_l)) * dt + abs(c.mdot * dt * c.gv.h / c.gv.M)
        c.Q_ig, c.Q_li, c.m_ev, c.n_ev, c.H_evL, c.H_evG = Q_ig, Q_li, m_ev, n_ev, H_evL, H_evG

    # ================================================================ 6. zone balances
    def _zone_balances(self, c: StepContext) -> None:
        opt, m, G, Lz, dt = self.opt, self.m, self.G, self.Lz, self.dt
        src_is_gas = c.src_is_gas
        n_out = c.mdot * dt / c.Mv * c.yG
        n_out = np.minimum(n_out, 0.5 * (G.n if src_is_gas else Lz.n))
        H_o = float(n_out.sum()) * c.gv.h
        self.H_out += H_o
        self.released += c.mdot * dt
        G_n = G.n - (n_out if src_is_gas else 0.0) + c.n_ev
        G_H = G.H + (c.Q_wg + c.Q_ig) * dt - (H_o if src_is_gas else 0.0) + c.H_evG
        L_n = Lz.n - c.n_ev - (0.0 if src_is_gas else n_out)
        sink = False
        if opt.water_mode == "sink":
            sink = True
        elif opt.water_mode == "vf" and m.iw is not None and c.liq_present:
            m_hc = float(np.delete(Lz.n, m.iw) @ np.delete(m.Mw, m.iw))
            sink = Lz.n[m.iw] > 0 and m_hc < 2.0
        Q_liq_in = c.Q_wl - c.Q_li + c.Q_rad_l
        if sink:
            self.Q_sink += Q_liq_in * dt
            Q_liq_in = 0.0
        L_H = Lz.H + Q_liq_in * dt - c.H_evL - (0.0 if src_is_gas else H_o)
        c.G_n, c.G_H, c.L_n, c.L_H = G_n, G_H, L_n, L_H
        c.VG0, c.VL0, c.P0s = G.V, Lz.V, self.P

    # ================================================================ 7. pressure
    # Zones that were single-phase last step use a fast single-phase solve during the
    # iteration; one stability test afterwards switches a zone to a full flash (and
    # re-solves P) only if a new phase has appeared.
    def _zone_flash(self, c: StepContext, n, H, Vold, Pt, prev, full, root, split_prev=None):
        m = self.m
        N = n.sum()
        if N < 1e-3:
            return None, 0.0
        h = (H + Vold * (Pt - c.P0s)) / N
        x = n / N
        if m.iw is not None and x[m.iw] >= 1.0 - 1e-12:
            rw = m.water_PH(Pt, h, T_guess=prev.T if prev is not None else None)
            return rw, N * rw.v
        r = None
        if not full and prev is not None and len(prev.phases) == 1 \
                and prev.phases[0].name != "aqueous":       # water pools use IAPWS (full flash)
            r = fast_PH_single(m, x, Pt, h, prev.T, root)
        elif prev is not None and len(prev.phases) > 1:
            r = fast_PH_multi(m, x, Pt, h, prev)          # same phase set, no stability
        elif full and split_prev is not None:
            r = fast_PH_multi(m, x, Pt, h, split_prev)    # zone split last step too
        if r is None:
            r = m.flash_PH(Pt, h, x, T_guess=prev.T if prev is not None else 300.0, init=prev)
        return r, N * r.v

    def _volume_residual(self, c: StepContext, Pt: float) -> float:
        rg, vg = self._zone_flash(c, c.G_n, c.G_H, c.VG0, Pt, self.G.r, c.full["G"], "V",
                                  self.last_split["G"])
        rl, vl = self._zone_flash(c, c.L_n, c.L_H, c.VL0, Pt, c.prevL, c.full["L"], "L",
                                  self.last_split["L"]) if c.L_n.sum() > 1e-3 else (None, 0.0)
        c.cache[Pt] = (rg, rl)
        return vg + vl - self.V

    def _solve_P(self, c: StepContext, P_start: float) -> float:
        sat, V = self.sat, self.V
        Pa, fa = P_start, self._volume_residual(c, P_start)
        slope = sat.get("dVdP")
        Pb_ = Pa - fa / slope if slope else (Pa * (1.0 - 1e-4) if fa < 0 else Pa * (1.0 + 1e-4))
        if not (0.5 * Pa < Pb_ < 2.0 * Pa):
            Pb_ = Pa * (1.0 - 1e-4) if fa < 0 else Pa * (1.0 + 1e-4)

        def safe(Pt, P_good):
            # a trial pressure far from the solution can put a zone where no flash
            # solution exists; back off towards the last good pressure
            for _ in range(12):
                try:
                    return Pt, self._volume_residual(c, Pt)
                except (RuntimeError, ValueError, ZeroDivisionError, FloatingPointError):
                    Pt = 0.5 * (Pt + P_good)
            return Pt, self._volume_residual(c, Pt)

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

    def _unstable(self, r, root: str) -> bool:
        m, P = self.m, self.P
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

    def _solve_pressure(self, c: StepContext, step: int) -> None:
        c.full = dict(self.sticky)       # zones that needed a full flash last step start full
        c.prevL = self.Lz.r if not self.Lz.empty else None
        c.cache = {}
        self.P = self._solve_P(c, self.P)
        rg, rl = c.cache[self.P]
        redo = False
        for key, r, root in (("G", rg, "V"), ("L", rl, "L")):
            if r is not None and r.kind == "fast1" and self._unstable(r, root):
                c.full[key] = True
                redo = True
        if redo:
            self.P = self._solve_P(c, self.P)
            rg, rl = c.cache[self.P]
        for key, r in (("G", rg), ("L", rl)):
            # stay in full-flash mode while the zone keeps splitting; retry fast mode
            # every 20 steps so a zone that has settled goes back to the cheap path
            self.sticky[key] = bool(r is not None and len(r.phases) > 1) or \
                (c.full[key] and step % 20 != 0)
            self.last_split[key] = r if (r is not None and len(r.phases) > 1) else None
        c.G_H = c.G_H + c.VG0 * (self.P - c.P0s)
        c.L_H = c.L_H + c.VL0 * (self.P - c.P0s)
        c.rg, c.rl = rg, rl

    # ================================================================ 8. phase transfer
    # The zone that loses a phase keeps the other phase(s) exactly (same P, T and phase
    # composition), so its state is built directly; only a zone that RECEIVES material
    # needs a new (single-phase, fast) solve.
    def _phase_transfer(self, c: StepContext) -> None:
        m, P, Lz, names = self.m, self.P, self.Lz, self.names
        G_n, G_H, L_n, L_H, rg, rl = c.G_n, c.G_H, c.L_n, c.L_H, c.rg, c.rl

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
            rG = (fast_PH_single(m, xg, P, G_H / G_n.sum(), Tg0, "V") if rG is not None
                  else None) or m.flash_PH(P, G_H / G_n.sum(), xg, T_guess=Tg0, init=rG)
        G = Zone("gas", G_n, G_H, rG) if (rG is not None and G_n.sum() > 1e-3) else None
        if L_n.sum() > 1e-3:
            if recvL or rL is None:
                xl, hl_ = L_n / L_n.sum(), L_H / L_n.sum()
                if rL is not None and len(rL.phases) == 1 and rL.phases[0].name == "liquid":
                    rL = fast_PH_single(m, xl, P, hl_, rL.T, "L") or \
                        m.flash_PH(P, hl_, xl, T_guess=rL.T, init=rL)
                else:
                    rL = m.flash_PH(P, hl_, xl, T_guess=Lz.T, init=rL)
            Lz = Zone("liq", L_n, L_H, rL)
        else:
            Lz = Zone("liq", np.zeros(len(names)), 0.0, G.r if G is not None else rG)
        if G is None:                       # gas zone vented / compressed away
            G = Zone("gas", np.zeros(len(names)), 0.0, Lz.r)
        self.G, self.Lz = G, Lz

    # ================================================================ 9. vanishing zones
    def _merge_vanishing_zones(self) -> None:
        m, P, names = self.m, self.P, self.names
        G, Lz = self.G, self.Lz
        # gas zone vented away / compressed out: one dense zone remains
        if not G.empty and not Lz.empty and (G.mass < max(1.0, 5e-3 * self.m_gas0)
                                             or Lz.V > (1.0 - 1e-3) * self.V):
            L_n2, L_H2 = Lz.n + G.n, Lz.H + G.H
            rL2 = m.flash_PH(P, L_H2 / L_n2.sum(), L_n2 / L_n2.sum(), T_guess=Lz.T, init=None)
            Lz = Zone("liq", L_n2, L_H2, rL2)
            G = Zone("gas", np.zeros(len(names)), 0.0, rL2)
            self.last_split["G"], self.sticky["G"] = None, False
        # boil-dry: a vanishing liquid zone joins the gas zone
        if not Lz.empty and (Lz.mass < max(1.0, 5e-3 * self.m_liq0)
                             or self.geom.level(Lz.V) < 5e-3):
            G_n, G_H = G.n + Lz.n, G.H + Lz.H
            rG = m.flash_PH(P, G_H / G_n.sum(), G_n / G_n.sum(), T_guess=G.T, init=None)
            G = Zone("gas", G_n, G_H, rG)
            Lz = Zone("liq", np.zeros(len(names)), 0.0, G.r)
            self.last_split["L"], self.sticky["L"] = None, False
        self.G, self.Lz = G, Lz

    # ================================================================ 10. wall areas
    def _reweight_wall(self) -> None:
        """Wall areas follow the level (energy-conserving re-weighting)."""
        frac, cols = self.frac, self.cols
        f_new = self.geom.wetted_perimeter_fraction(self._level()) if not self.Lz.empty else 0.0
        df = f_new - frac["wet"]
        if abs(df) > 1e-9:
            src, dst = ("dry", "wet") if df > 0 else ("wet", "dry")
            a = abs(df)
            if frac[dst] + a > 0:
                # energy-conserving mixing node by node: e(T) = integral of cp dT
                e_mix = (self._e_of_T(cols[dst].T) * frac[dst] + self._e_of_T(cols[src].T) * a) \
                    / (frac[dst] + a)
                cols[dst].T = np.interp(e_mix, self.E_TAB, self.T_TAB)
            frac[dst] += a
            frac[src] = max(frac[src] - a, 0.0)

    def _check_finite(self, c: StepContext) -> None:
        """Stop at the first non-finite state with diagnostics."""
        cols, G, P, info = self.cols, self.G, self.P, self.info
        if not (np.all(np.isfinite(cols["dry"].T)) and np.all(np.isfinite(cols["wet"].T))
                and np.isfinite(G.T) and np.isfinite(P) and np.isfinite(G.H)):
            raise FloatingPointError(
                f"non-finite state at t={c.time}: P={P}, G.T={G.T}, G.H={G.H}, G phases="
                f"{G.r.phase_names}, L phases={self.Lz.r.phase_names}, h_dry={info['h_dry']}, "
                f"Q_wg={info['Q_wg']}, dryT={cols['dry'].T[[0, -1]]}, "
                f"wetT={cols['wet'].T[[0, -1]]}, frac={self.frac}, yG={np.round(c.yG, 4)}")


def simulate(case: dict, opt: VesselFireOptions, mat: SteelTable) -> tuple[pd.DataFrame, dict]:
    """Run one vessel-in-fire case; returns (time series, meta)."""
    return VesselFireModel(case, opt, mat).run()
