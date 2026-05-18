#!/usr/bin/env python3
"""
==========================================================================
 1D TRANSIENT HEAT TRANSFER — LNG TANK INSULATION
 DUAL CASE: Liquid LNG Contact vs Vapor Space Contact

 INSULATION MODEL: Char-in-place (wire net + bolt retention)
   Based on Kawasaki GC-1 specification:
   - Panels fixed by fastening bolts (M6 Al alloy, 120mm)
   - Wire net reinforcement (0.62mm galvanized iron, 5mm mesh)
   - Char skeleton retained in place after decomposition
   - Conservative: radiation through char pores at high T

 CONSERVATIVE ASSUMPTIONS:
   - Char k higher than intact foam (porous carbon)
   - Radiation through char increases k at high temperatures
   - No credit for AL-PET reflectivity after PET melts
   - Decomposition endotherm ignored (conservative — less energy absorbed)
   - Properties chosen to OVERESTIMATE heat transfer through stack

 Layer stack (fire → LNG):
   AL-PET (0.2mm) → PUF (100mm) → PRF (100mm) → Aluminum (30mm)
==========================================================================
"""

import numpy as np
from collections import defaultdict
import os

# ============================================================
# CONSTANTS
# ============================================================
SIGMA = 5.67e-8
T_AMBIENT = 20.0
T_LNG = -163.0
EPS_STEEL = 0.8
GAP_WIDTH = 1.7

# LNG properties (pure methane, from Kawasaki GC-1 spec)
H_FG_LNG = 511e3          # J/kg (GC-1: 511 kJ/kg)
RHO_LNG_LIQUID = 425.0    # kg/m³ (GC-1: 425 kg/m³)
CP_LNG_VAPOR = 2200.0     # J/(kg·K) at ~-161.5°C
RHO_LNG_VAPOR = 1.8       # kg/m³

# Boundary condition parameters
H_VAPOR = 10.0            # W/m²K natural gas convection

# Vapor space geometry
VAPOR_VOLUME = 500.0      # m³
VAPOR_AREA = 200.0        # m²

# ============================================================
# BELTEMP FILE PATH — CHANGE THIS TO YOUR LOCAL PATH
# ============================================================
BELTEMP_FILE = "FAHTS_outer_dome_beltemp.fem"

# ============================================================
# LOAD STEEL TEMPERATURE
# ============================================================
def load_steel_temperature(beltemp_file):
    load_cases, data = [], defaultdict(dict)

    # --- Read file ---
    with open(beltemp_file, 'r') as f:
        for line in f:
            s = line.strip()

            if s.startswith("LCASETIM"):
                p = s.split()
                load_cases.append((int(p[1]), float(p[2])))

            elif s.startswith("BELTEMP"):
                p = s.split()
                data[int(p[1])][int(p[2])] = float(p[3])

    # Sort load cases by time
    load_cases.sort(key=lambda x: x[1])
    lc_nums = [lc[0] for lc in load_cases]
    times_min = [lc[1] for lc in load_cases]

    # ============================================================
    # FORCE ELEMENT SELECTION
    # ============================================================
    best_eid = 475

    # Optional safety check
    if best_eid not in data[lc_nums[0]]:
        raise ValueError(f"Element {best_eid} not found in BELTEMP data")

    # ============================================================
    # BUILD TEMPERATURE HISTORY
    # ============================================================
    times_sec = [0.0]
    temps = [20.0]

    T = 20.0
    for i, lc in enumerate(lc_nums):
        T += data[lc].get(best_eid, 0.0)
        times_sec.append(times_min[i] * 60.0)
        temps.append(T)

    return np.array(times_sec), np.array(temps), best_eid
# ============================================================
# MATERIAL PROPERTIES — CONSERVATIVE, CHAR-IN-PLACE
# ============================================================

# --- AL-PET ---

def k_alpet(T):
    """AL-PET conductivity.
    Conservative: after PET melts (260°C), lose credit for delamination resistance.
    After Al melts (660°C), only wire-net-held char residue remains.
    """
    points = [(-163, 260),(20, 235),(400, 220),(600, 210),(650, 205)]
    if T <= points[0][0]:
        return points[0][1]
    for i in range(len(points)-1):
        T1, k1 = points[i]
        T2, k2 = points[i+1]
        if T1 <= T <= T2:
            return k1 + (T - T1) * (k2 - k1) / (T2 - T1)
    # Sharp melt drop (narrow region)
    if T < 700:
        return 205 + (T-650)/50 * (95 - 205)
    return 95 + (T-700)/200 * 5


def cp_alpet(T):
    cp0 = 984.1
    if T < -100:
        return cp0 * 0.75
    if T < 0:
        return cp0 * (0.75 + (T+100)*0.0025)
    if T < 240:
        return cp0
    # PET decomposition (cp decreases)
    if T < 320:
        return cp0 + (T-240)/80 * (cp0*0.9 - cp0)
    # Char + Al mixture
    if T < 650:
        return cp0 * 0.9
    # Transition to Al-dominated
    if T < 700:
        return cp0*0.9 + (T-650)/50 * (800.0 - cp0*0.9)
    return 800.0

def rho_alpet(T):
    if T < 260:  return 1177.5
    if T < 350:  return 1177.5 * 0.60              # PET decomposed
    if T < 660:  return 1177.5 * 0.50              # Char + foil
    return 1177.5 * 0.20                           # Foil melted, char remains (wire net holds)


# --- PUF ---
def k_puf(T, alpha=0, dx=0.001):
    """PUF conductivity — char-in-place model.
    
    Char stays in place (wire net + bolts per GC-1 spec).
    Char has higher k than intact foam.
    At high T, radiation through char pores adds to effective k (CONSERVATIVE).
    
    k_char = 0.15 W/mK base (literature for PU char)
    k_rad_char = 4σ ε T³ × d_pore (radiation through ~1mm pores in char)
    
    Conservative: we use the HIGHER of (k_degrading_foam, k_char+radiation)
    """
    k0 = 0.0204
    #k_char_base = 0.15      # W/mK at room temperature
    d_pore = 0.002           # 2mm mean pore size in char (conservative: larger pores)
     
    # Known anchor points
    T1, k1 = -165, 0.0145
    T2, k2 = 24, 0.0204
    T3, k3 = 300, k0 * 3.0 
    
    if T<= 600:
    	k_char_base = 0.17
    elif T<=1200:
    	k_char_base = 0.9
    else:
    	k_char_base = 2.7
    
    
    # Interpolation
    if T <= T1:
        ki = k1
    elif T < T2:
        ki = k1 + (T - T1) * (k2 - k1) / (T2 - T1)
    elif T < T3:
        ki = k2 + (T - T2) * (k3 - k2) / (T3 - T2)
    else:
        ki = k3
        
    if alpha <= 0:
        return ki
    
    # Char conductivity (with radiation through pores — CONSERVATIVE)
    T_K = max(T + 273.15, 1.0)
    k_rad = 4 * SIGMA * 0.9 * T_K**3 * d_pore  # Radiation through char pores
    k_char = k_char_base + k_rad                 # Total char k
    
    # Blend intact and char
    # Use linear blend — conservative (series would give lower k)
    k_eff = ki * (1.0 - alpha) + k_char * alpha
    
    return k_eff



def cp_puf(T):
    """PUF specific heat. 
    Conservative: NO decomposition endotherm (removing the cp spike at 300°C).
    This means less energy is absorbed during decomposition → faster heat penetration.
    """
    cp0 = 1400.0
    T1 = -163
    cp1 = 450.0
    T_mid = 20
    cp_mid = 1400.0 
    if T < T1:
        return cp1
    if T <= T_mid:
        # linear rise from -163 to 20
        return cp1 + (T - T1) * (cp_mid - cp1) / (T_mid - T1)
    if T <= 300:
        return cp_mid
    return cp0 * 0.8



def rho_puf(T, alpha=0):
    """PUF density. Char stays in place but loses 70% mass as gas.
    Wire net retains the char skeleton — volume unchanged.
    """
    rho_foam = 41.4
    rho_char = rho_foam * 0.30    # 30% mass retained as char
    return rho_foam * (1.0 - alpha) + rho_char * alpha


# --- PRF ---
def k_prf(T, alpha=0, dx=0.001):
    """PRF conductivity — char-in-place model.
    PRF (phenolic resin foam) from GC-1: novolak type with cellulosic fibers.
    PRF char is denser and more structured than PUF char.
    k_char_PRF = 0.12 W/mK (slightly lower than PUF char).
    """
    k0 = 0.0349
    #k_char_base = 0.12
    d_pore = 0.002

    if T<= 600:
    	k_char_base = 0.17
    elif T<=1200:
    	k_char_base = 0.9
    else:
    	k_char_base = 2.7
    
     
    # Known anchor points
    T1, k1 = -175, 0.0116
    T2, k2 = 23, 0.0349
    T3, k3 = 425, 0.12
    
    # Interpolation
    if T <= T1:
        ki = k1
    elif T < T2:
        ki = k1 + (T - T1) * (k2 - k1) / (T2 - T1)
    elif T < T3:
        ki = k2 + (T - T2) * (k3 - k2) / (T3 - T2)
    else:
        ki = k3
    
    if alpha <= 0:
        return ki
    
    T_K = max(T + 273.15, 1.0)
    k_rad = 4 * SIGMA * 0.9 * T_K**3 * d_pore
    k_char = k_char_base + k_rad
    
    return ki * (1.0 - alpha) + k_char * alpha




def cp_prf(T):
    """PRF specific heat. Conservative: no decomposition endotherm."""
    cp0 = 1500.0
    T1 = -163
    cp1 = 400.0
    T_mid = 20
    cp_mid = 1500.0 
    if T < T1:
        return cp1
    if T <= T_mid:
        # linear rise from -163 to 20
        return cp1 + (T - T1) * (cp_mid - cp1) / (T_mid - T1)
    if T <= 400:
        return cp_mid
    return cp0 * 0.75


def rho_prf(T, alpha=0):
    """PRF density. Char retained by wire net."""
    rho_foam = 25.0
    rho_char = rho_foam * 0.30
    return rho_foam * (1.0 - alpha) + rho_char * alpha


# --- Aluminum ---
def k_aluminum(T):
    k0 = 120.0
    if T < -200:  return k0 * 0.58
    if T < -163:  return k0 * (0.58 + (T+200)/37*0.07)
    if T < 0:     return k0 * (0.65 + (T+163)/163*0.35)
    if T < 300:   return k0 * (1.0 + T/300*0.08)
    return k0 * 1.08

def cp_aluminum(T):
    points = [(-163, 472), (20, 900), (660, 1200), (800, 1100)]   
    if T <= points[0][0]:
        return points[0][1]  
    for i in range(len(points)-1):
        T1, cp1 = points[i]
        T2, cp2 = points[i+1]
        if T1 <= T <= T2:
            return cp1 + (T - T1) * (cp2 - cp1) / (T2 - T1)
    return points[-1][1]


def rho_aluminum(T): return 2650.0


# ============================================================
# MESH
# ============================================================
def build_mesh():
    layers = [
        ("AL-PET", 0.0002, 4,   k_alpet, cp_alpet, rho_alpet, None, None),
        ("PUF",    0.100,  100, k_puf,   cp_puf,   rho_puf,   300.0, 80.0),
        ("PRF",    0.100,  100, k_prf,   cp_prf,   rho_prf,   425.0, 150.0),
        ("Al",     0.030,  30,  k_aluminum, cp_aluminum, rho_aluminum, None, None),
    ]
    x, dx, kf, cpf, rhof, dt_arr, st_arr, lid_arr = [], [], [], [], [], [], [], []
    names = []
    x_pos, lid = 0.0, 0
    for name, thick, nc, kfn, cpfn, rhofn, dtemp, stemp in layers:
        cdx = thick / nc
        for i in range(nc):
            x.append(x_pos + (i+0.5)*cdx)
            dx.append(cdx)
            kf.append(kfn); cpf.append(cpfn); rhof.append(rhofn)
            dt_arr.append(dtemp); st_arr.append(stemp); lid_arr.append(lid)
        names.append(name)
        x_pos += thick
        lid += 1
    return (np.array(x), np.array(dx), kf, cpf, rhof,
            dt_arr, st_arr, np.array(lid_arr), names)


# ============================================================
# AIR GAP
# ============================================================
def air_gap_flux(T_steel, T_alpet):
    Ts_K = T_steel + 273.15
    Ta_K = T_alpet + 273.15
    # Conservative emissivity: after PET melts (260°C), quickly lose reflectivity
    # Don't wait until 660°C — PET substrate gone, foil wrinkles, ε rises
    '''
    if T_alpet < 150:     eps_a = 0.1
    elif T_alpet < 300:   eps_a = 0.1 + (T_alpet-150)/150 * 0.4   # 0.1→0.5 (conservative)
    elif T_alpet < 500:   eps_a = 0.5 + (T_alpet-300)/200*0.30   # 0.5→0.8
    else:                 eps_a = 0.90
    '''
    eps_a = 0.9
    denom = 1.0/EPS_STEEL + 1.0/eps_a - 1.0
    q_rad = SIGMA * (Ts_K**4 - Ta_K**4) / denom
    # Convection
    T_mean = 0.5*(T_steel + T_alpet)
    k_air = 0.026 + 6e-5*max(T_mean, 0)
    dT = abs(T_steel - T_alpet)
    beta = 1.0 / (max(T_mean,1)+273.15)
    Ra = max(9.81*beta*dT*GAP_WIDTH**3/(15e-6*22e-6), 1.0)
    Nu = max(1.0, 0.18*Ra**0.29)
    q_conv = Nu*k_air/GAP_WIDTH*(T_steel - T_alpet)
    return q_rad + q_conv


# ============================================================
# BOILING CURVE
# ============================================================
def h_boiling_LNG(T_wall, T_sat=-163.0):
    """Pool boiling curve for LNG (methane). Returns (h, regime)."""
    '''
    dT = T_wall - T_sat
    if dT <= 0:      h, regime = 150.0, "subcooled"
    elif dT < 3:     h, regime = 150.0 + dT/3.0*100.0, "nat_conv"
    elif dT < 10:
        frac = (dT-3)/(10-3)
        h, regime = 250.0 + frac**1.5*750.0, "nucl_onset"
    elif dT < 25:
        frac = (dT-10)/(25-10)
        h, regime = 1000.0 + frac*1500.0, "nucleate"
    elif dT < 40:
        frac = (dT-25)/(40-25)
        h, regime = 2500.0 - frac*1500.0, "trans_early"
    elif dT < 80:
        frac = (dT-40)/(80-40)
        h, regime = 1000.0 - frac*700.0, "transition"
    elif dT < 120:
        frac = (dT-80)/(120-80)
        h, regime = 300.0 - frac*220.0, "trans_late"
    else:
        h, regime = 80.0 + min(dT-120,200)*0.05, "film"
    '''
    h = 2000
    regime = "strong nuclate boiling"
    return h, regime


# ============================================================
# STEADY STATE
# ============================================================
def solve_steady_state(x, dx, k_funcs, layer_ids, layer_names):
    n = len(x)
    T_est = np.linspace(T_AMBIENT, T_LNG, n)
    R_cells = np.zeros(n)
    k_vals = np.zeros(n)
    for i in range(n):
        ki = k_funcs[i](T_est[i]) if 'alpha' not in k_funcs[i].__code__.co_varnames else k_funcs[i](T_est[i], 0, dx[i])
        R_cells[i] = dx[i] / ki
        k_vals[i] = ki
    R_total = np.sum(R_cells)
    q_ss = (T_AMBIENT - T_LNG) / R_total
    T_ss = np.zeros(n)
    R_c = 0.0
    for i in range(n):
        R_c += R_cells[i]/2
        T_ss[i] = T_AMBIENT - q_ss * R_c
        R_c += R_cells[i]/2
    # Refine
    for i in range(n):
        ki = k_funcs[i](T_ss[i]) if 'alpha' not in k_funcs[i].__code__.co_varnames else k_funcs[i](T_ss[i], 0, dx[i])
        R_cells[i] = dx[i] / ki
        k_vals[i] = ki
    R_total = np.sum(R_cells)
    q_ss = (T_AMBIENT - T_LNG) / R_total
    T_ss2 = np.zeros(n)
    R_c = 0.0
    for i in range(n):
        R_c += R_cells[i]/2
        T_ss2[i] = T_AMBIENT - q_ss * R_c
        R_c += R_cells[i]/2

    # Verbose output
    print(f"\n  BOUNDARY CONDITIONS:")
    print(f"    Left  (x=0): T = {T_AMBIENT}°C (ambient, no fire)")
    print(f"    Right (x=L): T = {T_LNG}°C (LNG boiling point)")
    print(f"    Governing eq: d/dx[k·dT/dx] = 0  →  q = const through all layers")
    print(f"\n  THERMAL RESISTANCE BUDGET:")
    print(f"    {'Layer':<12s} {'Thickness':>10s} {'k [W/mK]':>12s} {'R [m²K/W]':>12s} {'% of total':>10s}")
    print(f"    {'-'*58}")
    for lid, name in enumerate(layer_names):
        mask = layer_ids == lid
        R_layer = np.sum(R_cells[mask])
        k_avg = np.mean(k_vals[mask])
        t_layer = np.sum(dx[mask])
        pct = R_layer / R_total * 100
        print(f"    {name:<12s} {t_layer*1000:>8.2f}mm {k_avg:>10.4f}   {R_layer:>10.4f}   {pct:>8.1f}%")
    print(f"    {'-'*58}")
    print(f"    {'TOTAL':<12s} {np.sum(dx)*1000:>8.2f}mm {'':>12s} {R_total:>10.4f}   {'100.0':>8s}%")
    print(f"\n  STEADY-STATE SOLUTION:")
    print(f"    R_total = {R_total:.4f} m²K/W")
    print(f"    q_ss = ({T_AMBIENT}-({T_LNG}))/{R_total:.4f} = {q_ss:.3f} W/m²")
    print(f"\n  TEMPERATURE DISTRIBUTION:")
    print(f"    {'Location':<30s} {'x [mm]':>10s} {'T [°C]':>10s}")
    print(f"    {'-'*52}")
    for lid, name in enumerate(layer_names):
        mask = np.where(layer_ids == lid)[0]
        for label, idx in [("start", mask[0]), ("mid", mask[len(mask)//2]), ("end", mask[-1])]:
            print(f"    {name+' '+label:<30s} {x[idx]*1000:>10.3f} {T_ss2[idx]:>10.2f}")
    print(f"\n  INTERFACE TEMPERATURES:")
    print(f"    AL-PET outer:   {T_ss2[0]:>8.2f}°C")
    print(f"    AL-PET / PUF:   {0.5*(T_ss2[3]+T_ss2[4]):>8.2f}°C")
    print(f"    PUF / PRF:      {0.5*(T_ss2[103]+T_ss2[104]):>8.2f}°C")
    print(f"    PRF / Al:       {0.5*(T_ss2[203]+T_ss2[204]):>8.2f}°C")
    print(f"    Al inner:       {T_ss2[-1]:>8.2f}°C")
    print(f"\n  TEMPERATURE DROP PER LAYER:")
    for lid, name in enumerate(layer_names):
        mask = np.where(layer_ids == lid)[0]
        dT = T_ss2[mask[0]] - T_ss2[mask[-1]]
        print(f"    {name:<12s}: ΔT = {dT:>8.2f}°C  ({T_ss2[mask[0]]:>8.2f} → {T_ss2[mask[-1]]:>8.2f}°C)")
    print(f"\n  THERMAL MASS PER LAYER (ρ·cp·thickness):")
    layer_props = {"AL-PET": (1177.5, 984.1, 0.0002), "PUF": (41.4, 1400.0, 0.100),
                   "PRF": (25.0, 1500.0, 0.100), "Al": (2650.0, 900.0, 0.030)}
    total_tm = 0
    for name in layer_names:
        rho, cp, t = layer_props[name]
        tm = rho * cp * t
        total_tm += tm
        print(f"    {name:<12s} {tm:>10.1f} J/(m²K)")
    print(f"    {'TOTAL':<12s} {total_tm:>10.1f} J/(m²K)")

    return T_ss2, q_ss, R_total


# ============================================================
# TRANSIENT SOLVER
# ============================================================
def solve_transient(x, dx, k_funcs, cp_funcs, rho_funcs, decomp_temps,
                    service_temps, layer_ids, T_init, steel_times, steel_temps,
                    case_name="liquid", t_end=14400, dt=0.25):
    n = len(x)
    T = T_init.copy()
    alpha = np.zeros(n)

    if case_name == "liquid":
        h_right = 150.0
        T_right = T_LNG
    else:
        h_right = H_VAPOR
        T_right = T_LNG

    T_vapor = T_LNG
    m_vapor = RHO_LNG_VAPOR * VAPOR_VOLUME
    P_vapor = 101325.0
    boiloff_cumulative = 0.0
    boiloff_rate = 0.0
    boil_regime = "subcooled"

    # Storage: save every second
    n_seconds = int(t_end) + 1
    T_all = np.zeros((n_seconds, n))
    T_all[0] = T.copy()
    # Extra columns: T_steel, q_fire, eps, q_lng, h_right, boiloff_rate, boiloff_cum, T_vapor, P_vapor, n_dead
    extra_all = np.zeros((n_seconds, 11))
    T_s0 = np.interp(0, steel_times, steel_temps)
    extra_all[0] = [T_s0, 0, 0.05, 0, 150, 0, 0, T_LNG, 101.325, 0,0]
    next_save = 1

    # Tracking for per-minute print
    boiloff_rate_hist = []
    vapor_hist = []
    al_temp_hist = []
    q_into_lng_hist = []

    n_steps = int(t_end / dt)
    print(f"\n  [{case_name.upper()}] Solving: {n_steps} steps, dt={dt}s, saving every 1s")
    print(f"  Decomposition model: CHAR-IN-PLACE (wire net + bolt retention)")
    print(f"  Burning rates: PUF 28mm/min (0.467/s), PRF 22mm/min (0.367/s)")

    for step in range(1, n_steps + 1):
        t = step * dt
        T_steel = np.interp(t, steel_times, steel_temps)
        q_fire = air_gap_flux(T_steel, T[0])

        # Update liquid BC
        if case_name == "liquid":
            T_wall = T[-1]
            h_right, boil_regime = h_boiling_LNG(T_wall, T_LNG)
        elif case_name == "vapor":
            T_right = T_vapor

        # Assemble A·T_new = b
        A_mat = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            has_alpha = 'alpha' in k_funcs[i].__code__.co_varnames
            ki = k_funcs[i](T[i], alpha[i], dx[i]) if has_alpha else k_funcs[i](T[i])
            cpi = cp_funcs[i](T[i])
            has_alpha_r = 'alpha' in rho_funcs[i].__code__.co_varnames
            rhoi = rho_funcs[i](T[i], alpha[i]) if has_alpha_r else rho_funcs[i](T[i])

            mass = rhoi * cpi * dx[i] / dt
            K_left, K_right_c = 0.0, 0.0

            if i > 0:
                ha = 'alpha' in k_funcs[i-1].__code__.co_varnames
                kim1 = k_funcs[i-1](T[i-1], alpha[i-1], dx[i-1]) if ha else k_funcs[i-1](T[i-1])
                R_half = dx[i]/(2*ki) + dx[i-1]/(2*kim1)
                K_left = 1.0/R_half if R_half > 0 else 0

            if i < n-1:
                ha = 'alpha' in k_funcs[i+1].__code__.co_varnames
                kip1 = k_funcs[i+1](T[i+1], alpha[i+1], dx[i+1]) if ha else k_funcs[i+1](T[i+1])
                R_half = dx[i]/(2*ki) + dx[i+1]/(2*kip1)
                K_right_c = 1.0/R_half if R_half > 0 else 0

            A_mat[i,i] = mass + K_left + K_right_c
            if i > 0:   A_mat[i,i-1] = -K_left
            if i < n-1: A_mat[i,i+1] = -K_right_c
            b[i] = mass * T[i]

            if i == 0:
                b[i] += q_fire
            if i == n-1:
                A_mat[i,i] += h_right
                b[i] += h_right * T_right

        T_new = np.linalg.solve(A_mat, b)

        # Update decomposition using spec-sheet burning rates
        # PUF: 28 mm/min = 0.467 mm/s → rate per 1mm cell = 0.467/s
        # PRF: 22 mm/min = 0.367 mm/s → rate per 1mm cell = 0.367/s
        for i in range(n):
            if decomp_temps[i] is not None and alpha[i] < 1.0:
                if T_new[i] > service_temps[i]:
                    burn_rate = 0.467 if layer_ids[i] == 1 else 0.367
                    if T_new[i] > decomp_temps[i]:
                        rate = burn_rate
                    else:
                        frac = (T_new[i] - service_temps[i]) / (decomp_temps[i] - service_temps[i])
                        rate = burn_rate * frac**2
                    alpha[i] = min(1.0, alpha[i] + rate * dt)

        # Heat flux into LNG
        q_into_lng = h_right * (T_new[-1] - T_right)

        if case_name == "liquid":
            boiloff_rate = max(0, q_into_lng) / H_FG_LNG
            boiloff_cumulative += boiloff_rate * dt
        else:
            boiloff_rate = 0
            Q_tot = max(0, q_into_lng) * VAPOR_AREA
            T_vapor += Q_tot * dt / (m_vapor * CP_LNG_VAPOR)
            P_vapor = 101325.0 * (T_vapor+273.15) / (T_LNG+273.15)

        T = T_new

        # Emissivity at surface
        T_surf = T[0]
        if T_surf < 260:     eps_now = 0.05
        elif T_surf < 400:   eps_now = 0.05 + (T_surf-260)/(400-260)*0.15
        elif T_surf < 660:   eps_now = 0.20 + (T_surf-400)/(660-400)*0.30
        elif T_surf < 680:   eps_now = 0.50 + (T_surf-660)/20*0.40
        else:                eps_now = 0.90

        # Save every second
        t_sec = int(round(t))
        if t_sec == next_save and next_save < n_seconds:
            puf_dec = np.mean(alpha[4:104])
            prf_dec = np.mean(alpha[104:204])
            T_all[next_save] = T.copy()
            extra_all[next_save] = [T_steel, q_fire, eps_now, q_into_lng, h_right,
                                     boiloff_rate*1000, boiloff_cumulative*1000,
                                     T_vapor, P_vapor/1000, puf_dec*100, prf_dec*100]
            next_save += 1

        # Per-minute tracking + print
        if step % int(60/dt) == 0:
            t_min = t / 60.0
            puf_dec = np.mean(alpha[4:104])*100
            prf_dec = np.mean(alpha[104:204])*100
            if case_name == "liquid":
                boiloff_rate_hist.append((t_min, boiloff_rate, boiloff_cumulative))
                extra = (f"  h={h_right:.0f}[{boil_regime}]"
                         f"  boil={boiloff_rate*1000:.3f}g/(m²s)  cum={boiloff_cumulative*1000:.1f}g/m²")
            else:
                vapor_hist.append((t_min, T_vapor, P_vapor/1000))
                extra = f"  T_vap={T_vapor:.2f}°C  P={P_vapor/1000:.2f}kPa"
            al_temp_hist.append((t_min, T_new[204], T_new[-1]))
            q_into_lng_hist.append((t_min, q_into_lng))

            # Char front position
            puf_chars = sum(1 for i in range(4,104) if alpha[i] > 0.5)
            prf_chars = sum(1 for i in range(104,204) if alpha[i] > 0.5)

            print(f"    t={t_min:5.1f}min: T_steel={T_steel:7.1f}°C  ε={eps_now:.3f}"
                  f"  q_fire={q_fire:8.1f}W/m²  T_ALPET={T[0]:7.1f}°C"
                  f"  T_PUF_srf={T[4]:7.1f}°C  T_PRF_srf={T[104]:7.1f}°C"
                  f"  T_Al_out={T[204]:7.1f}°C  T_Al_in={T[-1]:7.1f}°C"
                  f"  q_LNG={q_into_lng:7.1f}W/m²"
                  f"  PUF_dec={puf_dec:5.1f}% ({puf_chars}mm charred)"
                  f"  PRF_dec={prf_dec:5.1f}% ({prf_chars}mm charred)"
                  f"{extra}")

    T_all = T_all[:next_save]
    extra_all = extra_all[:next_save]

    return {
        'T_all': T_all, 'extra_all': extra_all,
        'alpha': alpha,
        'boiloff_hist': boiloff_rate_hist, 'vapor_hist': vapor_hist,
        'al_temp_hist': al_temp_hist, 'q_into_lng_hist': q_into_lng_hist,
        'case': case_name,
        'boiloff_total': boiloff_cumulative if case_name=="liquid" else 0,
        'T_vapor_final': T_vapor if case_name=="vapor" else T_LNG,
        'P_vapor_final': P_vapor/1000 if case_name=="vapor" else 101.3,
    }


# ============================================================
# EXCEL OUTPUT
# ============================================================
def write_excel(res_liq, res_vap, x, dx, layer_ids, output_path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    n_cells = len(x)
    x_mm = x * 1000

    hdr_font = Font(name='Arial', bold=True, size=9, color='FFFFFF')
    data_font = Font(name='Arial', size=9)
    center = Alignment(horizontal='center')

    fills = {0: PatternFill('solid', fgColor='CC6600'),    # AL-PET
             1: PatternFill('solid', fgColor='2E7D32'),    # PUF
             2: PatternFill('solid', fgColor='00695C'),    # PRF
             3: PatternFill('solid', fgColor='4527A0')}    # Al
    extra_fill = PatternFill('solid', fgColor='B71C1C')
    time_fill = PatternFill('solid', fgColor='1F4E79')

    layer_names = {0: "AL-PET", 1: "PUF", 2: "PRF", 3: "Al"}
    extra_names = ["T_steel_C", "q_fire_Wm2", "eps_ALPET", "q_LNG_Wm2",
                   "h_right_Wm2K", "boiloff_g_m2_s", "boiloff_cum_g_m2",
                   "T_vapor_C", "P_vapor_kPa", "PUF_decomp_pct", "PRF_decomp_pct"]

    wb = Workbook()

    for ci, (res, sheet_name) in enumerate([
        (res_liq, "Liquid"),
        (res_vap, "Vapor"),
    ]):
        T_data = res['T_all']
        extra = res['extra_all']
        n_times = len(T_data)

        if ci == 0:
            ws = wb.active
            ws.title = sheet_name
        else:
            ws = wb.create_sheet(sheet_name)

        print(f"  Writing sheet: {sheet_name} ({n_times} rows × {n_cells+12} cols)...")

        # Row 1: Layer group headers
        ws.cell(row=1, column=1, value="Time").font = hdr_font
        ws.cell(row=1, column=1).fill = time_fill
        ws.cell(row=1, column=2, value="").font = hdr_font

        col = 3
        prev_lid = -1
        for i in range(n_cells):
            lid = layer_ids[i]
            if lid != prev_lid:
                cnt = int(np.sum(layer_ids == lid))
                ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col+cnt-1)
                c = ws.cell(row=1, column=col)
                c.value = f"{layer_names[lid]} ({cnt} cells)"
                c.font = hdr_font
                c.fill = fills[lid]
                c.alignment = center
                prev_lid = lid
            col += 1

        ecol = 3 + n_cells
        ws.merge_cells(start_row=1, start_column=ecol, end_row=1, end_column=ecol+10)
        c = ws.cell(row=1, column=ecol)
        c.value = "Boundary & Tracking"
        c.font = hdr_font; c.fill = extra_fill; c.alignment = center

        # Row 2: Column headers
        ws.cell(row=2, column=1, value="Time [s]").font = hdr_font
        ws.cell(row=2, column=1).fill = time_fill
        ws.cell(row=2, column=2, value="Time [min]").font = hdr_font
        ws.cell(row=2, column=2).fill = time_fill

        for i in range(n_cells):
            lid = layer_ids[i]
            c = ws.cell(row=2, column=3+i)
            c.value = f"{layer_names[lid]}_{i} ({x_mm[i]:.2f}mm)"
            c.font = Font(name='Arial', bold=True, size=7, color='FFFFFF')
            c.fill = fills[lid]
            c.alignment = center

        for j, en in enumerate(extra_names):
            c = ws.cell(row=2, column=ecol+j)
            c.value = en; c.font = hdr_font; c.fill = extra_fill; c.alignment = center

        # Row 3: dx
        ws.cell(row=3, column=1, value="dx [mm]").font = Font(name='Arial', bold=True, size=9)
        for i in range(n_cells):
            ws.cell(row=3, column=3+i, value=round(dx[i]*1000, 3)).font = data_font

        # Data rows
        for ti in range(n_times):
            row = 4 + ti
            ws.cell(row=row, column=1, value=ti).font = data_font
            ws.cell(row=row, column=2, value=round(ti/60.0, 4)).font = data_font
            for i in range(n_cells):
                c = ws.cell(row=row, column=3+i)
                c.value = round(float(T_data[ti, i]), 2)
                c.font = data_font
                c.number_format = '0.00'
            for j in range(extra.shape[1]):
                c = ws.cell(row=row, column=ecol+j)
                c.value = round(float(extra[ti, j]), 4)
                c.font = data_font

            if ti % 300 == 0 and ti > 0:
                print(f"    t={ti}s ({ti/60:.0f}min) written")

        # Column widths
        ws.column_dimensions['A'].width = 9
        ws.column_dimensions['B'].width = 10
        for j in range(10):
            ws.column_dimensions[get_column_letter(ecol+j)].width = 16
        ws.freeze_panes = 'C4'

    # Info sheet
    ws_info = wb.create_sheet("Info", 0)
    info = [
        ("1D Transient Heat Transfer — LNG Tank Thermal Protection", ""),
        ("Insulation Model", "CHAR-IN-PLACE (wire net + bolt retention per Kawasaki GC-1)"),
        ("Conservative", "YES — radiation through char pores, no decomposition endotherm"),
        ("", ""),
        ("MESH", f"{len(x)} cells: 4 AL-PET + 100 PUF + 100 PRF + 30 Al"),
        ("Total thickness", "230.20 mm"),
        ("PUF/PRF cell size", "1.0 mm"),
        ("AL-PET cell size", "0.05 mm"),
        ("", ""),
        ("BURNING RATES (from spec sheet)", ""),
        ("PUF", "28 mm/min (ASTM D-1692-68)"),
        ("PRF", "22 mm/min (ISO 3582)"),
        ("", ""),
        ("LNG PROPERTIES (Kawasaki GC-1)", ""),
        ("Temperature", "-161.5°C (using -163°C in model)"),
        ("Density", "425 kg/m³"),
        ("Latent heat", "511 kJ/kg"),
        ("Vapor pressure", "106 kPaA"),
        ("", ""),
        ("Liquid sheet", "h = boiling curve (150-2500 W/m²K), T_LNG fixed at -163°C"),
        ("Vapor sheet", "h = 10 W/m²K, T_vapor rises with heat input"),
    ]
    for r, (a, b) in enumerate(info, 1):
        ws_info.cell(row=r, column=1, value=a).font = Font(name='Arial', size=10, bold=(b=="" and a!=""))
        ws_info.cell(row=r, column=2, value=b).font = Font(name='Arial', size=10)
    ws_info.column_dimensions['A'].width = 40
    ws_info.column_dimensions['B'].width = 70

    print(f"\n  Saving workbook...")
    wb.save(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("  1D TRANSIENT HEAT TRANSFER — LNG TANK INSULATION")
    print("  Model: CHAR-IN-PLACE (wire net + bolt retention per Kawasaki GC-1 spec)")
    print("  Conservative: radiation through char pores, no decomposition endotherm, early ε degradation")
    print("  Case 1: Liquid LNG contact (dynamic boiling curve)")
    print("  Case 2: Vapor space contact (h=10 W/m²K, T rises)")
    print("=" * 100)

    steel_times, steel_temps, hot_eid = load_steel_temperature(BELTEMP_FILE)
    print(f"\n  Steel element {hot_eid}: peak {np.max(steel_temps):.1f}°C "
          f"at t={steel_times[np.argmax(steel_temps)]/60:.0f} min")

    print(f"\n  STEEL DOME TEMPERATURE HISTORY:")
    print(f"    {'t [min]':>8s} {'T [°C]':>10s}")
    for i in range(len(steel_times)):
        print(f"    {steel_times[i]/60:>6.1f}   {steel_temps[i]:>8.1f}")

    x, dx, kf, cpf, rhof, dts, sts, lids, names = build_mesh()
    n = len(x)
    print(f"\n  Mesh: {n} cells (4 AL-PET + 100 PUF + 100 PRF + 30 Al)")

    # Steady state
    print(f"\n{'='*100}")
    print(f"  STEADY-STATE (initial condition, no fire)")
    print(f"{'='*100}")
    T_ss, q_ss, R_total = solve_steady_state(x, dx, kf, lids, names)

    # Case 1: Liquid
    print(f"\n{'='*100}")
    print(f"  CASE 1: LIQUID LNG CONTACT")
    print(f"  Right BC: h = h_boiling(ΔT_wall), T_LNG = {T_LNG}°C")
    print(f"{'='*100}")
    res_liq = solve_transient(x, dx, kf, cpf, rhof, dts, sts, lids,
                               T_ss, steel_times, steel_temps,
                               case_name="liquid", t_end=14400, dt=0.25)

    # Case 2: Vapor
    print(f"\n{'='*100}")
    print(f"  CASE 2: VAPOR SPACE CONTACT")
    print(f"  Right BC: h = {H_VAPOR} W/m²K, T_vapor starts at {T_LNG}°C")
    print(f"{'='*100}")
    res_vap = solve_transient(x, dx, kf, cpf, rhof, dts, sts, lids,
                               T_ss, steel_times, steel_temps,
                               case_name="vapor", t_end=14400, dt=0.25)

    # Summary
    print(f"\n{'='*100}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'Parameter':<35s} {'LIQUID':>14s} {'VAPOR':>14s}")
    print(f"  {'-'*65}")

    for label, idx in [("AL-PET peak [°C]", 0), ("PUF surface peak [°C]", 4),
                        ("PRF surface peak [°C]", 104)]:
        vl = np.max(res_liq['T_all'][:, idx])
        vv = np.max(res_vap['T_all'][:, idx])
        print(f"  {label:<35s} {vl:>12.1f}°C {vv:>12.1f}°C")

    for label, idx in [("Al outer face peak [°C]", 204), ("Al inner face peak [°C]", -1)]:
        vl = np.max(res_liq['T_all'][:, idx])
        vv = np.max(res_vap['T_all'][:, idx])
        print(f"  {label:<35s} {vl:>12.1f}°C {vv:>12.1f}°C")
        if "inner" in label:
            print(f"  {'Al service limit exceeded?':<35s} {'YES' if vl>65 else 'NO':>14s} {'YES' if vv>65 else 'NO':>14s}")

    print(f"  {'PUF decomposition [%]':<35s} {np.mean(res_liq['alpha'][4:104])*100:>12.1f}% "
          f"{np.mean(res_vap['alpha'][4:104])*100:>12.1f}%")
    print(f"  {'PRF decomposition [%]':<35s} {np.mean(res_liq['alpha'][104:204])*100:>12.1f}% "
          f"{np.mean(res_vap['alpha'][104:204])*100:>12.1f}%")

    if res_liq['boiloff_hist']:
        print(f"\n  LIQUID — Boil-off:")
        print(f"    Total: {res_liq['boiloff_total']*1000:.1f} g/m²")
        bo = np.array(res_liq['boiloff_hist'])
        print(f"    Peak rate: {np.max(bo[:,1])*1000:.3f} g/(m²·s)")
        print(f"    LNG depth boiled: {res_liq['boiloff_total']/RHO_LNG_LIQUID*1e6:.1f} μm")

    if res_vap['vapor_hist']:
        print(f"\n  VAPOR — Vapor space:")
        print(f"    T_vapor final: {res_vap['T_vapor_final']:.1f}°C (rise: {res_vap['T_vapor_final']-T_LNG:.1f}°C)")
        print(f"    P_vapor final: {res_vap['P_vapor_final']:.1f} kPa (rise: {res_vap['P_vapor_final']-101.3:.1f} kPa)")

    # Excel output
    output_path = "temperature_results.xlsx"
    print(f"\n  Writing Excel: {output_path}")
    write_excel(res_liq, res_vap, x, dx, lids, output_path)


if __name__ == "__main__":
    main()
