"""
twophase_physics.py - two-phase (gas / hydrocarbon liquid / free water) physics
for the vfpy vessel model.

All physics here is taken from the open literature (sources cited at each
function). Nothing is derived from VessFire binaries or output. Property values
are passed in as floats / dicts so that any property source (CoolProp, the
Peng-Robinson package thermo_pr.py, tables) can be plugged in.

Contents
  1. GEOMETRY            level <-> volume, wetted area, interface area,
                         characteristic lengths, stacked water + oil layers
  2. WALL -> LIQUID      natural convection, nucleate boiling (Rohsenow,
                         Mostinski, Cooper, Gorenflo/VDI), CHF (Kutateladze-
                         Zuber, Lienhard-Dhir, Ivey-Morris subcooling), minimum
                         film boiling flux (Zuber/Berenson), film boiling
                         (Berenson, Bromley), transition boiling, full boiling
                         curve q(T_w) with dq/dT_w for implicit wall coupling
  3. WALL -> VAPOUR      free convection (Churchill-Chu etc.), Nusselt film
                         condensation (vertical wall / horizontal cylinder)
  4. INTERFACE           horizontal-surface free convection on both sides of
                         the free surface (Lloyd-Moran / Raithby-Hollands),
                         interface energy balance -> evaporation/condensation
  5. TWO-ZONE VESSEL     non-equilibrium gas + liquid zones sharing one
                         pressure (TwoZoneVessel), homogeneous-equilibrium
                         fallback (HomogeneousVessel), property-provider
                         protocol and a CoolProp pure-fluid reference provider.

Units: SI throughout (m, kg, s, K, Pa, J, W). Heat flux q > 0 means heat flows
from the wall INTO the fluid.

Property dict conventions (all SI):
  single-phase film/bulk props  : {"rho","cp","mu","k","beta"}   (beta = 1/v dv/dT)
  saturation props at pressure P: {"T_sat","rho_l","rho_v","h_l","h_v","h_fg",
                                   "sigma","mu_l","mu_v","k_l","k_v","cp_l","cp_v",
                                   "P","P_c","M"}  (M in kg/kmol)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

G_ACC = 9.80665
SIGMA_SB = 5.670374419e-8


# =============================================================================
# 1. GEOMETRY
# =============================================================================
#
# Vessel = cylindrical shell (inner radius R, straight length L) plus two heads
# of depth a:  flat (a = 0), hemispherical (a = R), 2:1 ellipsoidal (a = R/2),
# or any semi-ellipsoid depth a given explicitly.
#
# Horizontal vessel, liquid depth h measured from the bottom of the shell:
#   shell segment area   A_seg(h) = R^2 acos(1 - h/R) - (R - h) sqrt(2Rh - h^2)
#   shell volume         V_shell  = A_seg * L
#   both heads           V_heads  = pi a h^2 (3R - h) / (3R)
#     (exact for semi-ellipsoidal heads with semi-axes R, R, a; hemisphere
#      a = R gives the spherical-cap formula; e.g. D. Jones, "Calculating Tank
#      Volume", 2003; Perry's Handbook 8th ed., Sec. 10 tank-volume formulas)
#   free-surface width   w(h) = 2 sqrt(h (2R - h))
#   interface area       A_i = w L + 2 * (pi/2)(w/2)(a w / (2R))       (= dV/dh)
#   wetted shell area    A_w,shell = 2 R acos(1 - h/R) L
#   wetted head area     A_w,head  = A_head * h / (2R)
#     (exact for a hemisphere: sphere zone area 2 pi R h is linear in h
#      (Archimedes); used as an approximation for ellipsoidal heads)
#
# Vertical vessel, depth h measured from the bottom of the lower head:
#   bottom head (0 <= h <= a):  V = pi R^2 h^2 (3a - h) / (3a^2)
#                               r(h)^2 = R^2 (1 - ((a - h)/a)^2)
#   shell   (a < h < a + L):    V = V_head + pi R^2 (h - a)
#   top head: by symmetry V(h) = V_tot - V_bottom(H_tot - h)
#   wetted head area from the surface-of-revolution integral, parametrised by
#   the polar angle t: z = a(1 - cos t), r = R sin t,
#     dA = 2 pi R sin t sqrt(a^2 sin^2 t + R^2 cos^2 t) dt
#
# The inverse (volume -> level) uses Newton's method with dV/dh = A_i and a
# bisection safeguard.


def _head_area(R: float, a: float) -> float:
    """Inner surface area of one semi-ellipsoidal head (oblate/prolate spheroid
    half-surface), a = head depth. Exact closed forms (e.g. CRC Standard
    Mathematical Tables, spheroid surface area)."""
    if a <= 0:
        return math.pi * R**2
    if abs(a - R) < 1e-12 * R:
        return 2 * math.pi * R**2
    if a < R:   # oblate
        e = math.sqrt(1 - (a / R)**2)
        return math.pi * R**2 + math.pi * a**2 / (2 * e) * math.log((1 + e) / (1 - e))
    e = math.sqrt(1 - (R / a)**2)   # prolate
    return math.pi * R**2 + math.pi * R * a / e * math.asin(e)


def _cap_area_vertical(R: float, a: float, h: float) -> float:
    """Wetted area of a lower semi-ellipsoidal head filled to depth h (0..a)."""
    if a <= 0:
        return math.pi * R**2 if h > 0 else 0.0
    h = min(max(h, 0.0), a)
    t1 = math.acos(1 - h / a)
    f = lambda t: 2 * math.pi * R * math.sin(t) * math.sqrt(a**2 * math.sin(t)**2 + R**2 * math.cos(t)**2)
    return quad(f, 0.0, t1)[0]


@dataclass
class VesselGeometry:
    """Cylindrical vessel with semi-ellipsoidal (incl. hemispherical/flat) heads.

    D        : inner diameter [m]
    L        : straight (tangent-to-tangent) shell length [m]
    orientation : "horizontal" | "vertical"
    head     : "flat" | "hemispherical" | "ellipsoidal" (2:1) | float depth a [m]
    """
    D: float
    L: float
    orientation: str = "horizontal"
    head: object = "ellipsoidal"

    def __post_init__(self):
        self.R = 0.5 * self.D
        if isinstance(self.head, str):
            self.a = {"flat": 0.0, "hemispherical": self.R, "ellipsoidal": self.R / 2}[self.head]
        else:
            self.a = float(self.head)
        self.A_head = _head_area(self.R, self.a)
        self.A_total = 2 * math.pi * self.R * self.L + 2 * self.A_head
        self.V_total = math.pi * self.R**2 * self.L + 4.0 / 3.0 * math.pi * self.R**2 * self.a
        self.H = self.D if self.orientation == "horizontal" else self.L + 2 * self.a

    # ---------------------------------------------------------------- volume
    def volume(self, h: float) -> float:
        """Liquid volume below level h [m3]."""
        R, L, a = self.R, self.L, self.a
        h = min(max(h, 0.0), self.H)
        if self.orientation == "horizontal":
            A_seg = R**2 * math.acos(1 - h / R) - (R - h) * math.sqrt(max(2 * R * h - h * h, 0.0))
            V_heads = math.pi * a * h**2 * (3 * R - h) / (3 * R) if R > 0 else 0.0
            return A_seg * L + V_heads
        # vertical
        V_head = 2.0 / 3.0 * math.pi * R**2 * a
        def v_bottom(y):
            if y <= a:
                return math.pi * R**2 * y**2 * (3 * a - y) / (3 * a**2) if a > 0 else 0.0
            return V_head + math.pi * R**2 * (y - a)
        if h <= a + L:
            return v_bottom(h)
        return self.V_total - v_bottom(self.H - h)

    def interface_area(self, h: float) -> float:
        """Free-surface (gas-liquid) area at level h [m2]; equals dV/dh."""
        R, L, a = self.R, self.L, self.a
        if h <= 0 or h >= self.H:
            return 0.0
        if self.orientation == "horizontal":
            w = 2 * math.sqrt(h * (2 * R - h))
            return w * L + 2 * (math.pi / 2) * (w / 2) * (a * w / (2 * R))
        y = h if h <= a + L else self.H - h
        if y >= a:
            return math.pi * R**2
        return math.pi * R**2 * (1 - ((a - y) / a)**2)

    def wetted_area(self, h: float) -> float:
        """Inner wall area below level h [m2]."""
        R, L, a = self.R, self.L, self.a
        h = min(max(h, 0.0), self.H)
        if self.orientation == "horizontal":
            return 2 * R * math.acos(1 - h / R) * L + 2 * self.A_head * h / (2 * R)
        if h >= self.H:
            return self.A_total
        if h <= a:
            return _cap_area_vertical(R, a, h)
        if h <= a + L:
            return self.A_head + 2 * math.pi * R * (h - a)
        return self.A_total - _cap_area_vertical(R, a, self.H - h)

    def wetted_fraction(self, h: float) -> float:
        return self.wetted_area(h) / self.A_total

    def wetted_perimeter_fraction(self, h: float) -> float:
        """Fraction of the shell circumference below the level (horizontal),
        0/1 for vertical shells. = acos(1 - h/R)/pi."""
        if self.orientation != "horizontal":
            return 1.0 if h > self.a else 0.0
        return math.acos(1 - min(max(h, 0.0), self.D) / self.R) / math.pi

    # ---------------------------------------------------------------- inverse
    def level(self, V: float, tol: float = 1e-10) -> float:
        """Level for a liquid volume V (Newton with dV/dh = A_i, bisection guard)."""
        if V <= 0:
            return 0.0
        if V >= self.V_total:
            return self.H
        lo, hi = 0.0, self.H
        h = self.H * V / self.V_total
        for _ in range(100):
            f = self.volume(h) - V
            if abs(f) < tol * self.V_total:
                return h
            if f > 0:
                hi = h
            else:
                lo = h
            dVdh = self.interface_area(h)
            hn = h - f / dVdh if dVdh > 0 else 0.5 * (lo + hi)
            h = hn if lo < hn < hi else 0.5 * (lo + hi)
        return h

    # ---------------------------------------------------------------- lengths
    def interface_char_length(self, h: float) -> float:
        """Goldstein et al. (1973) / Lloyd & Moran (1974) horizontal-surface
        length scale L* = A / P of the free surface."""
        R, L, a = self.R, self.L, self.a
        if h <= 0 or h >= self.H:
            return 0.0
        if self.orientation == "horizontal":
            w = 2 * math.sqrt(h * (2 * R - h))
            Ai = self.interface_area(h)
            # rectangle w x L plus two half-ellipses (Ramanujan perimeter)
            b = a * w / (2 * R)
            ah = w / 2
            per_ell = math.pi * (3 * (ah + b) - math.sqrt((3 * ah + b) * (ah + 3 * b))) if (ah + b) > 0 else 0
            P = 2 * L + per_ell
            return Ai / P
        Ai = self.interface_area(h)
        return math.sqrt(Ai / math.pi) / 2.0      # disc: A/P = r/2

    def liquid_char_length(self, h: float) -> float:
        """Height of the heated wetted wall used for wall->liquid free
        convection (boundary layer runs up the wall to the free surface)."""
        return max(h, 1e-3)

    def gas_char_length(self, h: float) -> float:
        return max(self.H - h, 1e-3)


@dataclass
class LayerState:
    """Geometric description of stacked free-water + hydrocarbon layers."""
    h_water: float
    h_liquid: float            # top of hydrocarbon liquid (>= h_water)
    A_wet_water: float
    A_wet_oil: float
    A_wet_gas: float
    A_int_water_oil: float
    A_int_liquid_gas: float
    V_water: float
    V_oil: float
    V_gas: float


def layers(geom: VesselGeometry, V_water: float, V_oil: float) -> LayerState:
    """Water settles at the bottom, oil above it, gas on top (VessFire's
    published initial state uses a hydrocarbon level plus a free-water level)."""
    hw = geom.level(V_water)
    hl = geom.level(V_water + V_oil)
    Aw = geom.wetted_area(hw)
    Al = geom.wetted_area(hl)
    return LayerState(hw, hl, Aw, Al - Aw, geom.A_total - Al,
                      geom.interface_area(hw) if V_oil > 0 else 0.0,
                      geom.interface_area(hl),
                      V_water, V_oil, geom.V_total - V_water - V_oil)


def layers_from_levels(geom: VesselGeometry, h_water: float, h_liquid: float) -> LayerState:
    Vw = geom.volume(h_water)
    return layers(geom, Vw, geom.volume(h_liquid) - Vw)


# =============================================================================
# 2. WALL -> LIQUID HEAT TRANSFER
# =============================================================================

def rayleigh(props: dict, dT: float, L: float) -> tuple[float, float]:
    """Ra = g beta |dT| L^3 / (nu alpha), Pr. props at film temperature."""
    Pr = props["cp"] * props["mu"] / props["k"]
    Ra = (G_ACC * abs(props["beta"]) * abs(dT) * L**3 * props["rho"]**2 * props["cp"]
          / (props["mu"] * props["k"]))
    return Ra, Pr


def nu_free(Ra: float, Pr: float, config: str) -> float:
    """Free-convection Nusselt numbers.

    vertical_plate      Churchill & Chu (1975a), Int. J. Heat Mass Transfer 18:1323,
                        full range: Nu = {0.825 + 0.387 Ra^(1/6) / [1+(0.492/Pr)^(9/16)]^(8/27)}^2
    horizontal_cylinder Churchill & Chu (1975b), Int. J. Heat Mass Transfer 18:1049:
                        Nu = {0.60 + 0.387 Ra^(1/6) / [1+(0.559/Pr)^(9/16)]^(8/27)}^2
    mcadams_vertical    McAdams (1954): 0.59 Ra^1/4 (laminar), 0.10 Ra^1/3 (turbulent, Ra>1e9)
    evans_stefany       Evans & Stefany (1966), closed-container transient heating: 0.55 Ra^1/4
    hot_up              Lloyd & Moran (1974), J. Heat Transfer 96:443, heated surface facing
                        up / cooled facing down (unstable): 0.54 Ra^1/4 (Ra<1e7), 0.15 Ra^1/3
    hot_down            Raithby & Hollands (Handbook of Heat Transfer, 1998) / Incropera
                        eq. 9.32, heated facing down / cooled facing up (stable): 0.52 Ra^1/5
    """
    Ra = max(Ra, 0.0)
    if config == "vertical_plate":
        return (0.825 + 0.387 * Ra**(1 / 6) / (1 + (0.492 / Pr)**(9 / 16))**(8 / 27))**2
    if config == "horizontal_cylinder":
        return (0.60 + 0.387 * Ra**(1 / 6) / (1 + (0.559 / Pr)**(9 / 16))**(8 / 27))**2
    if config == "mcadams_vertical":
        return 0.10 * Ra**(1 / 3) if Ra > 1e9 else 0.59 * Ra**0.25
    if config == "evans_stefany":
        return 0.55 * Ra**0.25
    if config == "hot_up":
        return 0.54 * Ra**0.25 if Ra < 1e7 else 0.15 * Ra**(1 / 3)
    if config == "hot_down":
        return 0.52 * Ra**0.2
    raise ValueError(config)


def h_free(props: dict, dT: float, L: float, config: str = "vertical_plate") -> float:
    """Free-convection coefficient [W/m2K]; props at film temperature."""
    if abs(dT) < 1e-9 or L <= 0:
        dT = 1e-9 if abs(dT) < 1e-9 else dT
    Ra, Pr = rayleigh(props, dT, L)
    return nu_free(Ra, Pr, config) * props["k"] / L


def q_free(T_w: float, T_f: float, props: dict, L: float, config: str = "vertical_plate") -> float:
    return h_free(props, T_w - T_f, L, config) * (T_w - T_f)


# ------------------------------------------------------------------ nucleate
# Every correlation is returned as a function dT_sat -> q (W/m2).
# Correlations written as h = C q^m are inverted analytically:
#   q = h dT = C q^m dT  ->  q = (C dT)^(1/(1-m))

def nb_rohsenow(sat: dict, C_sf: float = 0.013, s: float | None = None, water: bool = False):
    """Rohsenow (1952), Trans. ASME 74:969:
       q = mu_l h_fg [g (rho_l - rho_v)/sigma]^(1/2) [cp_l dT / (C_sf h_fg Pr_l^s)]^3
       s = 1.0 for water, 1.7 for other fluids; C_sf surface/fluid constant
       (0.013 generic; Pioro 1999 tables). Returns q(dT)."""
    if s is None:
        s = 1.0 if water else 1.7
    Pr_l = sat["cp_l"] * sat["mu_l"] / sat["k_l"]
    A = sat["mu_l"] * sat["h_fg"] * math.sqrt(G_ACC * (sat["rho_l"] - sat["rho_v"]) / sat["sigma"])
    B = sat["cp_l"] / (C_sf * sat["h_fg"] * Pr_l**s)
    return lambda dT: A * (B * max(dT, 0.0))**3


def nb_mostinski(sat: dict):
    """Mostinski (1963), Teploenergetika 4:66 (as given in Collier & Thome 1994):
       h = 0.00417 q^0.7 Pc[kPa]^0.69 F(pr),  F = 1.8 pr^0.17 + 4 pr^1.2 + 10 pr^10
       Reduced-pressure correlation: needs only Pc - good for mixtures."""
    pr = sat["P"] / sat["P_c"]
    F = 1.8 * pr**0.17 + 4 * pr**1.2 + 10 * pr**10
    C = 0.00417 * (sat["P_c"] / 1e3)**0.69 * F
    m = 0.7
    return lambda dT: (C * max(dT, 0.0))**(1 / (1 - m))


def nb_cooper(sat: dict, Rp_um: float = 1.0, factor: float = 1.0):
    """Cooper (1984), Adv. Heat Transfer 16:157:
       h = 55 pr^(0.12 - 0.2 log10 Rp) (-log10 pr)^(-0.55) M^(-0.5) q^0.67
       Rp surface roughness [um] (1 um default), M [kg/kmol]; 'factor' = 1.7 is
       Cooper's suggestion for horizontal copper cylinders (1.0 for plates)."""
    pr = min(sat["P"] / sat["P_c"], 0.95)   # correlation undefined at pr >= 1 (complex)
    C = factor * 55 * pr**(0.12 - 0.2 * math.log10(Rp_um)) * (-math.log10(pr))**(-0.55) * sat["M"]**(-0.5)
    m = 0.67
    return lambda dT: (C * max(dT, 0.0))**(1 / (1 - m))


# VDI Heat Atlas (2010), chapter H2 (Gorenflo & Kenning), Table 1: reference
# heat transfer coefficients h0 [W/m2K] at pr0 = 0.1, q0 = 20 kW/m2, Ra0 = 0.4 um.
# Values transcribed from the published table; verify before production use.
GORENFLO_H0 = {"methane": 7000.0, "ethane": 4500.0, "propane": 4000.0,
               "n-butane": 3600.0, "isobutane": 3600.0, "n-pentane": 3400.0,
               "water": 5600.0, "nitrogen": 4500.0}


def nb_gorenflo(sat: dict, h0: float, Ra_um: float = 0.4, water: bool = False):
    """Gorenflo (1993) / VDI Heat Atlas H2:
       h/h0 = F_PF (q/q0)^n (Ra/Ra0)^(2/15)
       F_PF = 1.2 pr^0.27 + 2.5 pr + pr/(1 - pr);  n = 0.9 - 0.3 pr^0.3   (general)
       water: F_PF = 1.73 pr^0.27 + (6.1 + 0.68/(1-pr)) pr^2, n = 0.9 - 0.3 pr^0.15"""
    pr = sat["P"] / sat["P_c"]
    if water:
        F = 1.73 * pr**0.27 + (6.1 + 0.68 / (1 - pr)) * pr**2
        n = 0.9 - 0.3 * pr**0.15
    else:
        F = 1.2 * pr**0.27 + 2.5 * pr + pr / (1 - pr)
        n = 0.9 - 0.3 * pr**0.3
    C = h0 * F * (Ra_um / 0.4)**(2 / 15) / 20000.0**n
    return lambda dT: (C * max(dT, 0.0))**(1 / (1 - n))


# ------------------------------------------------------------------ CHF / MHF
def lambda_taylor(sat: dict) -> float:
    """Capillary (Laplace) length sqrt(sigma / (g (rho_l - rho_v)))."""
    return math.sqrt(sat["sigma"] / (G_ACC * (sat["rho_l"] - sat["rho_v"])))


def q_chf(sat: dict, geometry: str = "large_plate", R_heater: float | None = None,
          dT_sub: float = 0.0) -> float:
    """Critical heat flux.
    Kutateladze (1948) / Zuber (1959):  q_max = C h_fg rho_v^1/2 [sigma g (rho_l - rho_v)]^1/4
      "zuber"        C = 0.131 (Zuber's value; conservative)
      "large_plate"  C = 0.149 (Lienhard & Dhir 1973, NASA CR-2270, large flat heaters)
      "cylinder"     Sun & Lienhard (1970), Int. J. HMT 13:1425, horizontal cylinder:
                     q_max / q_Z(0.131) = 0.89 + 2.27 exp(-3.44 sqrt(R')), R' = R/lambda >= 0.15
    Subcooling: Ivey & Morris (1962), UKAEA AEEW-R137:
      q_max,sub = q_max [1 + 0.1 (rho_l/rho_v)^0.75 cp_l dT_sub / h_fg]
    """
    base = sat["h_fg"] * math.sqrt(sat["rho_v"]) * (sat["sigma"] * G_ACC * (sat["rho_l"] - sat["rho_v"]))**0.25
    if geometry == "zuber":
        q = 0.131 * base
    elif geometry == "large_plate":
        q = 0.149 * base
    elif geometry == "cylinder":
        Rp = max(R_heater / lambda_taylor(sat), 0.15)
        q = 0.131 * base * (0.89 + 2.27 * math.exp(-3.44 * math.sqrt(Rp)))
    else:
        raise ValueError(geometry)
    if dT_sub > 0:
        q *= 1 + 0.1 * (sat["rho_l"] / sat["rho_v"])**0.75 * sat["cp_l"] * dT_sub / sat["h_fg"]
    return q


def q_min_film(sat: dict) -> float:
    """Minimum (Leidenfrost) heat flux, Zuber (1959) / Berenson (1961), large
    horizontal surface: q_min = 0.09 rho_v h_fg [sigma g (rho_l-rho_v)/(rho_l+rho_v)^2]^1/4"""
    rl, rv = sat["rho_l"], sat["rho_v"]
    return 0.09 * rv * sat["h_fg"] * (sat["sigma"] * G_ACC * (rl - rv) / (rl + rv)**2)**0.25


def h_film_boiling(sat: dict, dT: float, T_w: float, geometry: str = "berenson",
                   D: float | None = None, eps_w: float = 0.8) -> float:
    """Film boiling coefficient incl. radiation.
    berenson  Berenson (1961), J. Heat Transfer 83:351, large horizontal surface:
              h = 0.425 [k_v^3 rho_v g (rho_l-rho_v) h'_fg / (mu_v dT lambda)]^1/4,
              h'_fg = h_fg + 0.5 cp_v dT  (lambda = Taylor length)
    bromley   Bromley (1950), Chem. Eng. Prog. 46:221, horizontal cylinder diameter D:
              h = 0.62 [k_v^3 rho_v g (rho_l-rho_v) h'_fg / (mu_v D dT)]^1/4,
              h'_fg = h_fg + 0.4 cp_v dT
    Radiation across the vapour film (Bromley 1950): h = h_fb + 0.75 h_rad,
      h_rad = eps_w sigma (T_w^4 - T_sat^4)/(T_w - T_sat) (liquid taken as black).
    Vapour props (k_v, mu_v, cp_v, rho_v) are taken at saturation for simplicity;
    film-temperature evaluation can be supplied by passing a modified sat dict."""
    dT = max(dT, 1e-6)
    rl, rv = sat["rho_l"], sat["rho_v"]
    if geometry == "berenson":
        hfg = sat["h_fg"] + 0.5 * sat["cp_v"] * dT
        L, C = lambda_taylor(sat), 0.425
    else:
        hfg = sat["h_fg"] + 0.4 * sat["cp_v"] * dT
        L, C = D, 0.62
    h_fb = C * (sat["k_v"]**3 * rv * G_ACC * (rl - rv) * hfg / (sat["mu_v"] * dT * L))**0.25
    Ts = T_w - dT
    h_rad = eps_w * SIGMA_SB * (T_w**4 - Ts**4) / dT
    return h_fb + 0.75 * h_rad


# ------------------------------------------------------------------ boiling curve
@dataclass
class BoilingModel:
    """Settings for the wall->liquid boiling curve."""
    nucleate: str = "cooper"             # rohsenow | mostinski | cooper | gorenflo (see notes)
    chf: str = "zuber"                   # zuber | large_plate | cylinder
    film: str = "berenson"               # berenson | bromley
    nc_config: str = "vertical_plate"    # liquid free convection correlation
    blend_n: float = 3.0                 # Churchill-Usagi power for nc + nucleate
    dT_onb: float = 0.0                  # onset-of-boiling wall superheat [K]
    subcooled_chf: bool = True
    C_sf: float = 0.013                  # Rohsenow
    Rp_um: float = 1.0                   # Cooper
    h0_gorenflo: float | None = None     # Gorenflo reference h0
    water: bool = False
    eps_w: float = 0.8                   # wall emissivity (film boiling radiation)
    D: float | None = None               # heater diameter (bromley / cylinder CHF)

    def nucleate_fn(self, sat):
        if self.nucleate == "rohsenow":
            return nb_rohsenow(sat, self.C_sf, water=self.water)
        if self.nucleate == "mostinski":
            return nb_mostinski(sat)
        if self.nucleate == "cooper":
            return nb_cooper(sat, self.Rp_um)
        if self.nucleate == "gorenflo":
            return nb_gorenflo(sat, self.h0_gorenflo, water=self.water)
        raise ValueError(self.nucleate)


def boiling_flux(T_w: float, T_l: float, sat: dict, liq_props: dict, L_nc: float,
                 model: BoilingModel = BoilingModel()) -> tuple[float, str]:
    """Heat flux wall -> liquid pool [W/m2] and regime name.

    Regimes (e.g. Incropera & DeWitt ch. 10; Collier & Thome 1994 ch. 4):
      T_w - T_sat <= dT_onb                 : single-phase free convection to T_l
      dT_onb < dT_sat <= dT_CHF             : nucleate boiling, blended with free
                                              convection (partial/subcooled boiling),
                                              Churchill & Usagi (1972) asymptotic sum
                                              q = (q_nc^n + q_nb^n)^(1/n)
      dT_CHF < dT_sat < dT_MIN              : transition boiling, log-log interpolation
                                              between (dT_CHF, q_CHF) and (dT_MIN, q_MIN)
      dT_sat >= dT_MIN                      : film boiling (Berenson/Bromley + radiation)
    dT_CHF solves q_nb(dT) = q_CHF (analytic for the power-law forms), dT_MIN solves
    q_film(dT) = q_MIN. Hysteresis (the nucleate branch followed on heating) is not
    modelled - the curve is single valued.
    If T_w < T_l the flux is negative (liquid heats the wall) by free convection.
    """
    dT_nc = T_w - T_l
    q_nc = q_free(T_w, T_l, liq_props, L_nc, model.nc_config)
    dT_sat = T_w - sat["T_sat"]
    if dT_sat <= model.dT_onb:
        return q_nc, "free_convection"
    qnb_fn = model.nucleate_fn(sat)
    dT_sub = max(sat["T_sat"] - T_l, 0.0) if model.subcooled_chf else 0.0
    qmax = q_chf(sat, model.chf, (model.D or 1.0) / 2, dT_sub)
    # dT at CHF (monotone q_nb): bracket and solve
    dT_chf = _solve_monotone(lambda d: qnb_fn(d) - qmax, 1e-3, 500.0)
    if dT_sat <= dT_chf:
        q_nb = qnb_fn(dT_sat)
        n = model.blend_n
        q = (max(q_nc, 0.0)**n + q_nb**n)**(1 / n)
        return min(q, qmax) if q_nb < qmax else qmax, "nucleate"
    qmin = q_min_film(sat)
    f_film = lambda d: h_film_boiling(sat, d, sat["T_sat"] + d, model.film, model.D, model.eps_w) * d
    dT_min = _solve_monotone(lambda d: f_film(d) - qmin, 1e-2, 2000.0)
    if dT_min <= dT_chf * 1.05:            # near-critical: no distinct transition branch
        dT_min = dT_chf * 1.05
        qmin = min(qmin, f_film(dT_min))
    if dT_sat < dT_min:
        w = math.log(dT_sat / dT_chf) / math.log(dT_min / dT_chf)
        return math.exp((1 - w) * math.log(qmax) + w * math.log(qmin)), "transition"
    return f_film(dT_sat), "film"


def _solve_monotone(f, lo, hi):
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        return lo
    while fhi < 0 and hi < 1e5:
        hi *= 2
        fhi = f(hi)
    return brentq(f, lo, hi, xtol=1e-6)


def with_derivative(qfun: Callable[[float], float], T_w: float, dT: float = 0.02):
    """q(T_w) and dq/dT_w by central difference (the boiling curve is piecewise
    smooth; a centred step of 0.02 K is well inside every regime width)."""
    qp, qm = qfun(T_w + dT), qfun(T_w - dT)
    return qfun(T_w), (qp - qm) / (2 * dT)


def boiling_flux_and_derivative(T_w, T_l, sat, liq_props, L_nc, model=BoilingModel()):
    """(q, dq/dT_w, regime) for implicit coupling to the wall conduction solver."""
    f = lambda Tw: boiling_flux(Tw, T_l, sat, liq_props, L_nc, model)[0]
    q, dq = with_derivative(f, T_w)
    return q, dq, boiling_flux(T_w, T_l, sat, liq_props, L_nc, model)[1]


def linearised_bc(q0: float, dq: float, T_w0: float, h_min: float = 1.0):
    """Convert q(T_w) ~ q0 + dq (T_w - T_w0) into the (h_eff, T_ref) form
    q = h_eff (T_w - T_ref) expected by heat_transfer.WallColumn.step(T_gas=T_ref,
    h_in=h_eff). In transition boiling dq < 0; h_eff is floored at h_min for the
    tridiagonal solver's diagonal dominance (the flux at T_w0 stays exact)."""
    h = max(dq, h_min)
    return h, T_w0 - q0 / h


# =============================================================================
# 3. WALL -> VAPOUR:  free convection (above) and film condensation
# =============================================================================

def condensation_flux(T_w: float, sat: dict, geometry: str = "horizontal_cylinder",
                      L: float = 1.0) -> tuple[float, float, float]:
    """Laminar Nusselt (1916) film condensation of a saturated vapour on a wall
    colder than T_sat. Returns (q, dq/dT_w, h); q < 0 (heat flows fluid -> wall,
    sign convention: q > 0 is wall -> fluid).
      vertical_wall        h = 0.943 [g rho_l (rho_l - rho_v) k_l^3 h'_fg / (mu_l L dT)]^1/4
      horizontal_cylinder  h = 0.729 [ ...                               / (mu_l D dT)]^1/4
      h'_fg = h_fg + 0.68 cp_l dT   (Rohsenow 1956, subcooling of the film)
    Inside a horizontal vessel the condensate drains down the concave inner wall;
    the outside-tube constant 0.729 is used as the closest published analogue
    (Chato 1962 gives ~0.555 for stratified condensation inside tubes with low
    vapour velocity - a more conservative alternative).
    Limitations: laminar film (Re_film < ~30-1800), no non-condensable
    resistance (multi-component gases lower h markedly; Colburn-Hougen / Silver-
    Bell-Ghaly corrections are needed for mixtures), no wave enhancement."""
    dT = sat["T_sat"] - T_w
    if dT <= 0:
        return 0.0, 0.0, 0.0
    C = 0.943 if geometry == "vertical_wall" else 0.729
    rl, rv = sat["rho_l"], sat["rho_v"]

    def h_of(d):
        hfg = sat["h_fg"] + 0.68 * sat["cp_l"] * d
        return C * (G_ACC * rl * (rl - rv) * sat["k_l"]**3 * hfg / (sat["mu_l"] * L * d))**0.25

    h = h_of(dT)
    q = -h * dT
    e = 1e-3 * dT
    dq = (-(h_of(dT - e) * (dT - e)) + h_of(dT + e) * (dT + e)) / (2 * e)  # d(-q)/d(dT)=d q/dT_w
    return q, dq, h


# =============================================================================
# 4. VAPOUR-LIQUID INTERFACE
# =============================================================================
#
# The free surface is taken at the saturation temperature of the liquid at the
# common pressure, T_i = T_sat(P) (bubble point for mixtures), as in Haque et
# al. (1992b, Trans IChemE 70B:10), Speranza & Terenzi (2005), Mahgerefteh & Wong
# (1999, Comput. Chem. Eng. 23:1309) and Andreasen (HydDown, JOSS 2021).
# Each side exchanges heat with the surface by free convection over a horizontal
# surface of characteristic length L* = A/P:
#   gas side   : T_g > T_i  -> warm gas above cold surface: stable      -> "hot_down"
#                T_g < T_i  -> unstable                                  -> "hot_up"
#   liquid side: T_l > T_i  -> cold surface on top of warm liquid: unstable -> "hot_up"
#                T_l < T_i  -> warm surface on top of cold liquid: stable  -> "hot_down"
# Interface energy balance (no storage in the surface):
#   Q_l->i = h_li A_i (T_l - T_i),  Q_i->g = h_gi A_i (T_i - T_g)
#   mdot_evap = (Q_l->i - Q_i->g) / h_fg      (> 0 evaporation, < 0 condensation)
# Liquid energy loses Q_l->i + mdot h_l,sat ; gas gains Q_i->g + mdot h_v,sat.


def interface_exchange(T_g: float, T_l: float, sat: dict, gas_props: dict, liq_props: dict,
                       A_i: float, L_star: float, h_override: tuple | None = None) -> dict:
    """Heat and mass exchange across the free surface. Returns
    {"Q_li","Q_ig","mdot_evap","h_li","h_gi","T_i"} (W, W, kg/s, W/m2K)."""
    T_i = sat["T_sat"]
    if A_i <= 0 or L_star <= 0:
        return dict(Q_li=0.0, Q_ig=0.0, mdot_evap=0.0, h_li=0.0, h_gi=0.0, T_i=T_i)
    if h_override:
        h_li, h_gi = h_override
    else:
        h_gi = h_free(gas_props, T_g - T_i, L_star, "hot_down" if T_g > T_i else "hot_up")
        h_li = h_free(liq_props, T_l - T_i, L_star, "hot_up" if T_l > T_i else "hot_down")
    Q_li = h_li * A_i * (T_l - T_i)
    Q_ig = h_gi * A_i * (T_i - T_g)
    return dict(Q_li=Q_li, Q_ig=Q_ig, mdot_evap=(Q_li - Q_ig) / sat["h_fg"],
                h_li=h_li, h_gi=h_gi, T_i=T_i)


# =============================================================================
# 5. PROPERTY PROVIDER + TWO-ZONE / HOMOGENEOUS VESSEL
# =============================================================================

class PropertyProvider(Protocol):
    """Interface a thermodynamic package must offer (mass basis). `z` is the
    zone composition (mass-fraction vector) or None for a pure fluid.

    ph(P, h, z) -> {"T","rho","x","h_l","h_v","rho_l","rho_v","z_l","z_v"}
        x = vapour mass fraction of the equilibrium split; x <= 0 subcooled
        liquid, x >= 1 superheated vapour (values outside [0,1] are allowed and
        mean "single phase", by how much in enthalpy terms is (h-h_l)/h_fg).
    sat(P, z)  -> saturation dict (see module docstring), bubble point of z
    transport(P, T, phase, z) -> {"rho","cp","mu","k","beta"}   phase "liquid"|"gas"
    gas_gamma(P, T, z) -> cp/cv of the vapour (for the valve)
    """
    def ph(self, P, h, z=None) -> dict: ...
    def sat(self, P, z=None) -> dict: ...
    def transport(self, P, T, phase, z=None) -> dict: ...
    def gas_gamma(self, P, T, z=None) -> float: ...


class CoolPropPure:
    """Reference PropertyProvider for a pure fluid using CoolProp (testing only;
    production runs use the Peng-Robinson package)."""

    def __init__(self, fluid: str, backend: str = "HEOS"):
        import CoolProp.CoolProp as CP
        self.CP = CP
        self.AS = CP.AbstractState(backend, fluid)
        self.fluid = fluid
        self.P_c = self.AS.p_critical()
        self.T_c = self.AS.T_critical()
        self.M = self.AS.molar_mass() * 1e3
        self._sat_cache = {}

    def sat(self, P, z=None):
        key = round(P, 3)
        s = self._sat_cache.get(key)
        if s is not None:
            return s
        AS, CP = self.AS, self.CP
        AS.update(CP.PQ_INPUTS, P, 0.0)
        s = dict(P=P, P_c=self.P_c, M=self.M, T_sat=AS.T(), rho_l=AS.rhomass(), h_l=AS.hmass(),
                 mu_l=AS.viscosity(), k_l=AS.conductivity(), cp_l=AS.cpmass(),
                 sigma=AS.surface_tension())
        AS.update(CP.PQ_INPUTS, P, 1.0)
        s.update(rho_v=AS.rhomass(), h_v=AS.hmass(), mu_v=AS.viscosity(),
                 k_v=AS.conductivity(), cp_v=AS.cpmass())
        s["h_fg"] = s["h_v"] - s["h_l"]
        if len(self._sat_cache) > 20000:
            self._sat_cache.clear()
        self._sat_cache[key] = s
        return s

    def ph(self, P, h, z=None):
        s = self.sat(P)
        x = (h - s["h_l"]) / s["h_fg"]
        if 0.0 < x < 1.0:
            v = (1 - x) / s["rho_l"] + x / s["rho_v"]
            return dict(T=s["T_sat"], rho=1 / v, x=x, h_l=s["h_l"], h_v=s["h_v"],
                        rho_l=s["rho_l"], rho_v=s["rho_v"], z_l=None, z_v=None)
        AS, CP = self.AS, self.CP
        AS.update(CP.HmassP_INPUTS, h, P)
        return dict(T=AS.T(), rho=AS.rhomass(), x=x, h_l=s["h_l"], h_v=s["h_v"],
                    rho_l=s["rho_l"], rho_v=s["rho_v"], z_l=None, z_v=None)

    def transport(self, P, T, phase, z=None):
        AS, CP = self.AS, self.CP
        try:
            AS.specify_phase(CP.iphase_liquid if phase == "liquid" else CP.iphase_gas)
            AS.update(CP.PT_INPUTS, P, T)
            out = dict(rho=AS.rhomass(), cp=AS.cpmass(), mu=AS.viscosity(),
                       k=AS.conductivity(), beta=AS.isobaric_expansion_coefficient())
        except ValueError:
            out = None
        finally:
            AS.unspecify_phase()
        if out is None or not all(np.isfinite(list(out.values()))):
            AS.update(CP.PQ_INPUTS, P, 0.0 if phase == "liquid" else 1.0)
            out = dict(rho=AS.rhomass(), cp=AS.cpmass(), mu=AS.viscosity(),
                       k=AS.conductivity(), beta=AS.isobaric_expansion_coefficient())
        return out

    def gas_gamma(self, P, T, z=None):
        AS, CP = self.AS, self.CP
        try:
            AS.specify_phase(CP.iphase_gas)
            AS.update(CP.PT_INPUTS, P, T)
            g = AS.cpmass() / AS.cvmass()
        except ValueError:
            AS.update(CP.PQ_INPUTS, P, 1.0)
            g = AS.cpmass() / AS.cvmass()
        finally:
            AS.unspecify_phase()
        return g

    def uv(self, rho, u):
        """Homogeneous UV flash (for HomogeneousVessel)."""
        AS, CP = self.AS, self.CP
        AS.update(CP.DmassUmass_INPUTS, rho, u)
        Q = AS.Q()
        x = Q if 0.0 <= Q <= 1.0 else (1.0 if AS.T() > self.T_c or rho < AS.rhomass() else 0.0)
        return dict(P=AS.p(), T=AS.T(), x=x, h=AS.hmass())

    def saturated_state(self, T):
        AS, CP = self.AS, self.CP
        AS.update(CP.QT_INPUTS, 0.0, T)
        P, hl, rl = AS.p(), AS.hmass(), AS.rhomass()
        AS.update(CP.QT_INPUTS, 1.0, T)
        return dict(P=P, h_l=hl, rho_l=rl, h_v=AS.hmass(), rho_v=AS.rhomass())


# ------------------------------------------------------------------ valves
@dataclass
class Valve:
    """Orifice at elevation z_nozzle (m above vessel bottom). Gas: ideal-gas
    isentropic nozzle with real density (choked / subcritical), e.g. API 520
    Part I eqs. for gas. Liquid (level above nozzle): incompressible Bernoulli,
    G = Cd sqrt(2 rho_l dP) - no flashing in the nozzle (API 520 Annex C omega
    method / HEM should replace this for saturated liquid)."""
    d: float
    cd: float = 0.84
    z_nozzle: float | None = None       # None = top of vessel
    P_back: float = 101325.0
    open_time: float = 0.0
    set_pressure: float | None = None   # if given: opens when P >= set (simple PSV, no blowdown)

    def area(self):
        return math.pi * self.d**2 / 4

    def gas_mdot(self, P, rho, gamma):
        k = gamma
        r = self.P_back / P
        rc = (2 / (k + 1))**(k / (k - 1))
        if r <= rc:
            G = math.sqrt(k * P * rho * (2 / (k + 1))**((k + 1) / (k - 1)))
        else:
            G = math.sqrt(2 * P * rho * k / (k - 1) * (r**(2 / k) - r**((k + 1) / k))) if r < 1 else 0.0
        return self.cd * self.area() * G

    def liquid_mdot(self, P, rho_l):
        dP = max(P - self.P_back, 0.0)
        return self.cd * self.area() * math.sqrt(2 * rho_l * dP)


# ------------------------------------------------------------------ wall regions
@dataclass
class WallRegion:
    """Wall heat-transfer region (wetted or dry) represented by one WallColumn
    (heat_transfer.WallColumn, per metre of length) scaled to area A."""
    col: object
    A: float

    @property
    def scale(self):                    # equivalent length [m]
        return self.A / self.col.A_in


def transfer_wall_area(src: WallRegion, dst: WallRegion, dA: float):
    """Move area dA (>0) from src to dst when the level moves, mixing the moved
    wall's temperature profile into dst (area-weighted, node by node). Exactly
    energy conserving for constant cp; for T-dependent cp the error is second
    order in the temperature difference."""
    dA = min(dA, src.A)
    if dA <= 0:
        return
    dst.col.T = (dst.A * dst.col.T + dA * src.col.T) / (dst.A + dA)
    dst.A += dA
    src.A -= dA


# ------------------------------------------------------------------ zones
@dataclass
class Zone:
    m: float            # kg
    H: float            # J (total enthalpy)
    z: object = None    # composition (mass fractions) or None
    T: float = 0.0
    V: float = 0.0
    rho: float = 0.0
    x: float = 0.0

    @property
    def h(self):
        return self.H / self.m if self.m > 0 else float("nan")


@dataclass
class TwoZoneOptions:
    dt: float = 1.0
    boiling: BoilingModel = field(default_factory=BoilingModel)
    gas_wall_config: str = "vertical_plate"   # free convection gas <-> dry wall
    condensation: bool = True
    interface: bool = True
    relax_rainout: float = 1.0    # fraction of supersaturation condensed per step
    relax_boiloff: float = 1.0    # fraction of liquid-zone vapour released per step
    P_tol: float = 1e-9           # relative volume-residual tolerance


class TwoZoneVessel:
    """Non-equilibrium two-zone vessel: gas zone (g) and liquid zone (l), each
    with its own mass, enthalpy, composition and temperature, sharing one
    pressure P and the vessel volume V.

    State:   m_g, H_g, z_g, m_l, H_l, z_l, P (+ wall column temperatures)
    Balances over a step (open-system first law with boundary work between the
    zones; H = U + P V_zone):
        dm_g = (-mdot_out,g + mdot_evap - mdot_rain + mdot_boil - mdot_cond) dt
        dm_l = (-mdot_out,l - mdot_evap + mdot_rain - mdot_boil + mdot_cond) dt
        dU_k = dQ_k + sum(h dm)_k - P dV_k  <=>  dH_k = dQ_k + sum(h dm)_k + V_k dP
    Volume constraint:  V_g(P, h_g) + V_l(P, h_l) = V   ->  one scalar equation
    for the new pressure.

    Per-step algorithm (explicit transfer rates, implicit wall and pressure):
      1. At the old state: T_sat(P); wall -> liquid flux from the boiling curve,
         linearised (q0, dq/dTw) -> (h_eff, T_ref) for the wall solver; wall ->
         gas free convection h (or Nusselt condensation if T_w < T_sat(P) and
         the gas zone is at/near saturation); interface exchange; valve flow.
      2. Advance wall columns implicitly (backward Euler), obtain Q_wl, Q_wg
         from the new inner-surface temperatures (exactly what the wall lost).
      3. Explicit enthalpy/mass updates of both zones with those Q's and the
         mass transfers (interface evaporation, wall condensate, valve).
      4. Solve  F(P) = m_g/rho_g(P, h_g(P)) + m_l/rho_l(P, h_l(P)) - V = 0
         with H_k(P) = H_k* + V_k,old (P - P_old)  (Brent's method). Using the
         old zone volumes (which sum exactly to V) makes the total internal
         energy U = H_g + H_l - P V conserved to round-off.
      5. Phase-split each zone at (P, h_k): gas zone x_g < 1 -> rain-out of
         (1 - x_g) m_g at h_l,sat into the liquid; liquid zone x_l > 0 ->
         boil-off of x_l m_l at h_v,sat into the gas (relaxation factors < 1
         give finite-rate / metastable behaviour, e.g. retained bubbles =
         level swell). Transfers are at constant P and conserve H exactly.
      6. Update level, wetted / interface areas; move wall area between the
         wet and dry columns with energy-conserving mixing.
    Conservation checks: m_g + m_l + integral(mdot_out) = m_0 exactly;
    U_fluid + E_wall - integral(Q_outer) + integral(mdot h_out) = const.
    """

    def __init__(self, geom: VesselGeometry, props, T0: float, fill_frac: float,
                 wall_factory: Callable[[float], object], valve: Valve | None = None,
                 opt: TwoZoneOptions = TwoZoneOptions(), P0: float | None = None):
        """fill_frac = initial liquid volume fraction; zones start saturated at
        T0 (or liquid at T0 under pressure P0 if given - subcooled start).
        wall_factory(T0) must return a new heat_transfer.WallColumn."""
        self.g, self.pp, self.valve, self.opt = geom, props, valve, opt
        s = props.saturated_state(T0)
        P = s["P"] if P0 is None else P0
        V_l = fill_frac * geom.V_total
        V_g = geom.V_total - V_l
        if P0 is None:
            self.liq = Zone(V_l * s["rho_l"], V_l * s["rho_l"] * s["h_l"])
            self.gas = Zone(V_g * s["rho_v"], V_g * s["rho_v"] * s["h_v"])
        else:
            tl = props.transport(P, T0, "liquid")
            self.liq = Zone(V_l * tl["rho"], 0.0)
            # enthalpy of subcooled liquid: bisect on T via ph
            self.liq.H = self.liq.m * _h_at_PT(props, P, T0, "liquid")
            tg = props.transport(P, T0, "gas")
            self.gas = Zone(V_g * tg["rho"], V_g * tg["rho"] * _h_at_PT(props, P, T0, "gas"))
        self.P = P
        self.m0 = self.gas.m + self.liq.m
        self._evaluate()
        h = geom.level(self.liq.V)
        A_wet = geom.wetted_area(h)
        self.wet = WallRegion(wall_factory(T0), A_wet)
        self.dry = WallRegion(wall_factory(T0), geom.A_total - A_wet)
        self.t = 0.0
        self.m_out = 0.0
        self.H_out = 0.0
        self.Q_outer = 0.0
        self.U0 = self.U_fluid()
        self.E_wall0 = self.E_wall()
        self.rows = []
        self.record({})

    # ------------------------------------------------------------ helpers
    def _evaluate(self):
        """Zone states at (P, h_k); volumes."""
        for zn in (self.gas, self.liq):
            if zn.m <= 0:               # zone absent (e.g. liquid boiled dry)
                zn.m, zn.H, zn.V, zn.x = 0.0, 0.0, 0.0, 0.0
                zn.T, zn.rho = self.pp.sat(self.P)["T_sat"] if self.P < getattr(self.pp, "P_c", np.inf)                     else self.gas.T, float("nan")
                continue
            r = self.pp.ph(self.P, zn.h, zn.z)
            zn.T, zn.rho, zn.x = r["T"], r["rho"], r["x"]
            zn.V = zn.m / zn.rho

    def U_fluid(self):
        return self.gas.H + self.liq.H - self.P * self.g.V_total

    def E_wall(self):
        return sum(r.col.energy() * r.scale for r in (self.wet, self.dry))

    def level(self):
        return self.g.level(self.liq.V)

    # ------------------------------------------------------------ step
    def step(self):
        dt, g, pp, opt = self.opt.dt, self.g, self.pp, self.opt
        P0 = self.P
        sat = pp.sat(P0)
        gas, liq = self.gas, self.liq
        hlev = g.level(liq.V)
        info = {}

        # --- 1a wall -> liquid (boiling curve), linearised for the implicit wall
        Tw_wet = self.wet.col.T_inner
        lp = pp.transport(P0, 0.5 * (Tw_wet + liq.T), "liquid")
        liq_on = liq.m > 0
        if self.wet.A > 0 and liq_on:
            q0, dq, regime = boiling_flux_and_derivative(
                Tw_wet, liq.T, sat, lp, g.liquid_char_length(hlev), opt.boiling)
        else:
            q0, dq, regime = 0.0, 1.0, "none"
        h_wet, Tref_wet = linearised_bc(q0, dq, Tw_wet)
        info["regime"] = regime

        # --- 1b wall -> gas: free convection or condensation
        Tw_dry = self.dry.col.T_inner
        cond = False
        if opt.condensation and Tw_dry < sat["T_sat"] - 0.01 and gas.T < sat["T_sat"] + 1.0:
            qc, dqc, _ = condensation_flux(Tw_dry, sat, "horizontal_cylinder" if g.orientation == "horizontal"
                                           else "vertical_wall", g.D if g.orientation == "horizontal"
                                           else g.gas_char_length(hlev))
            h_dry, Tref_dry = linearised_bc(qc, dqc, Tw_dry)
            cond = True
        else:
            gpf = pp.transport(P0, 0.5 * (Tw_dry + gas.T), "gas")
            h_dry = h_free(gpf, Tw_dry - gas.T, g.gas_char_length(hlev), opt.gas_wall_config)
            Tref_dry = gas.T

        # --- 2 implicit wall step; outer heat for the energy audit
        t_mid = self.t + 0.5 * dt
        Q_w = {}
        for name, reg, h_in, Tref in (("wet", self.wet, h_wet, Tref_wet), ("dry", self.dry, h_dry, Tref_dry)):
            col = reg.col
            Ts_old = col.T_outer
            qo, dqo, _, _ = col.outer(Ts_old, t_mid)
            col.step(dt, t_mid, Tref, h_in)
            self.Q_outer += (qo + dqo * (col.T_outer - Ts_old)) * col.A_out * reg.scale * dt
            Q_w[name] = col.q_in * col.A_in * reg.scale          # W into fluid
        info.update(Q_wl=Q_w["wet"], Q_wg=Q_w["dry"])

        # --- 3 explicit zone updates
        dH_g = Q_w["dry"] * dt
        dH_l = Q_w["wet"] * dt
        if not liq_on:                  # residual wetted-wall area with no liquid
            dH_g, dH_l = dH_g + dH_l, 0.0
        dm_g = dm_l = 0.0
        if cond and Q_w["dry"] < 0:
            # film condensate: gas at h_g leaves, liquid at h_l,sat arrives, Q to wall
            m_c = -Q_w["dry"] * dt / max(gas.h - sat["h_l"], 1e3)
            m_c = min(m_c, 0.5 * gas.m)
            dH_g = -m_c * gas.h
            dm_g -= m_c
            dm_l += m_c
            dH_l += m_c * gas.h + Q_w["dry"] * dt   # = m_c h_l,sat (approximately)
            info["m_cond"] = m_c
        if opt.interface and liq_on and 0 < hlev < g.H:
            gp = pp.transport(P0, 0.5 * (gas.T + sat["T_sat"]), "gas")
            lpi = pp.transport(P0, 0.5 * (liq.T + sat["T_sat"]), "liquid")
            ex = interface_exchange(gas.T, liq.T, sat, gp, lpi, g.interface_area(hlev),
                                    g.interface_char_length(hlev))
            me = ex["mdot_evap"] * dt
            me = max(min(me, 0.2 * liq.m), -0.2 * gas.m)
            dH_l += -ex["Q_li"] * dt - me * sat["h_l"]
            dH_g += ex["Q_ig"] * dt + me * sat["h_v"]
            dm_l -= me
            dm_g += me
            info.update(Q_li=ex["Q_li"], Q_ig=ex["Q_ig"], mdot_evap=ex["mdot_evap"])
        # valve
        mdot = 0.0
        if self.valve is not None and self.t >= self.valve.open_time and \
                (self.valve.set_pressure is None or P0 >= self.valve.set_pressure):
            zn = g.H if self.valve.z_nozzle is None else self.valve.z_nozzle
            if liq_on and hlev >= zn:          # liquid at the nozzle
                mdot = self.valve.liquid_mdot(P0, liq.rho)
                mdot = min(mdot, 0.5 * liq.m / dt)
                dm_l -= mdot * dt
                dH_l -= mdot * dt * liq.h
                self.H_out += mdot * dt * liq.h
            else:
                gam = pp.gas_gamma(P0, gas.T)
                mdot = self.valve.gas_mdot(P0, gas.rho, gam)
                mdot = min(mdot, 0.5 * gas.m / dt)
                dm_g -= mdot * dt
                dH_g -= mdot * dt * gas.h
                self.H_out += mdot * dt * gas.h
            self.m_out += mdot * dt
        info["mdot"] = mdot

        gas.m += dm_g
        liq.m += dm_l
        gas.H += dH_g
        liq.H += dH_l
        Vg0, Vl0 = g.V_total - liq.V, liq.V      # old volumes, sum exactly V
        Hg_s, Hl_s = gas.H, liq.H

        # --- 4 pressure from the volume constraint
        def resid(P):
            V = 0.0
            for zn, Hs, V0 in ((gas, Hg_s, Vg0), (liq, Hl_s, Vl0)):
                if zn.m > 0:
                    V += zn.m / pp.ph(P, (Hs + V0 * (P - P0)) / zn.m, zn.z)["rho"]
            return V - g.V_total

        P = _solve_pressure(resid, P0)
        gas.H = Hg_s + Vg0 * (P - P0)
        liq.H = Hl_s + Vl0 * (P - P0)
        self.P = P
        self._evaluate()

        # --- 5 phase split at constant P (H conserving)
        s1 = pp.sat(P)
        if gas.x < 1.0 and P < getattr(pp, "P_c", np.inf):
            m_r = opt.relax_rainout * (1 - max(gas.x, 0.0)) * gas.m
            dH = gas.H if m_r >= gas.m else m_r * s1["h_l"]    # whole zone condensed
            gas.m -= m_r
            gas.H -= dH
            liq.m += m_r
            liq.H += dH
            info["m_rain"] = m_r
        if liq.m > 0 and liq.x > 0.0 and P < getattr(pp, "P_c", np.inf):
            m_b = opt.relax_boiloff * min(liq.x, 1.0) * liq.m
            dH = liq.H if m_b >= liq.m else m_b * s1["h_v"]    # whole zone evaporated
            liq.m -= m_b
            liq.H -= dH
            gas.m += m_b
            gas.H += dH
            info["m_boil"] = m_b
        self._evaluate()
        # zone disappearance: a vanishing liquid zone is merged into the gas zone
        # (H additive at constant P -> energy conserving); the liquid zone
        # re-appears through rain-out. A vanishing gas zone = liquid-full vessel.
        if 0 < liq.V < 1e-6 * g.V_total:
            gas.m += liq.m
            gas.H += liq.H
            liq.m = liq.H = 0.0
            self._evaluate()
        if gas.V < 1e-6 * g.V_total:
            raise RuntimeError("vessel liquid-full (gas zone vanished): hydraulic "
                               "overpressure - not modelled by the two-zone formulation")

        # --- 6 geometry / wall area bookkeeping
        A_wet_new = g.wetted_area(g.level(liq.V))
        dA = A_wet_new - self.wet.A
        if dA > 0:
            transfer_wall_area(self.dry, self.wet, dA)
        elif dA < 0:
            transfer_wall_area(self.wet, self.dry, -dA)
        self.t += dt
        self.record(info)

    # ------------------------------------------------------------ output
    def record(self, info):
        m_err = (self.gas.m + self.liq.m + self.m_out - self.m0) / self.m0
        E = self.U_fluid() + self.E_wall() + self.H_out - self.Q_outer
        E0 = self.U0 + self.E_wall0
        row = dict(t=self.t, P=self.P, T_g=self.gas.T, T_l=self.liq.T, m_g=self.gas.m,
                   m_l=self.liq.m, level=self.level(), V_l=self.liq.V,
                   Tw_wet=self.wet.col.T_inner, Tw_dry=self.dry.col.T_inner,
                   Tw_dry_out=self.dry.col.T_outer, A_wet=self.wet.A,
                   m_out=self.m_out, mass_err=m_err,
                   energy_err=(E - E0) / max(abs(self.Q_outer) + abs(self.H_out), 1.0))
        row.update(info)
        self.rows.append(row)

    def run(self, t_end):
        while self.t < t_end - 1e-9:
            self.step()
        import pandas as pd
        return pd.DataFrame(self.rows)


def _solve_pressure(resid, P0):
    """Root of the (monotonically decreasing) volume residual, bracketed by
    geometric expansion around the previous pressure, then Brent's method."""
    f0 = resid(P0)
    if f0 == 0.0:
        return P0
    fac = 1.002
    a, fa = P0, f0
    for _ in range(60):
        b = a * fac if f0 > 0 else a / fac     # F decreases with P
        fb = resid(b)
        if fb * f0 <= 0:
            lo, hi = (b, a) if b < a else (a, b)
            return brentq(resid, lo, hi, xtol=1e-9 * P0, rtol=1e-14, maxiter=200)
        a, fa = b, fb
        fac = fac**1.6
    raise RuntimeError("pressure bracket not found")


def _h_at_PT(props, P, T, phase):
    """Enthalpy at (P, T) via bisection on the provider's ph flash (keeps the
    provider interface minimal)."""
    s = props.sat(P)
    if phase == "liquid":
        lo, hi = s["h_l"] - 2e6, s["h_l"]
    else:
        lo, hi = s["h_v"], s["h_v"] + 3e6
    return brentq(lambda h: props.ph(P, h)["T"] - T, lo, hi, xtol=1e-3)


class HomogeneousVessel:
    """Homogeneous-equilibrium fallback: one zone, both phases at the same T
    (= T_sat when two-phase), UV flash each step. Wall -> fluid: boiling curve
    on the wetted wall with T_l = T_sat (no subcooling), free convection on the
    dry wall to the mixture temperature. Valve takes saturated vapour when the
    nozzle is above the level. Simplest robust model and the limit of
    TwoZoneVessel for infinite interface transfer."""

    def __init__(self, geom, props, T0, fill_frac, wall_factory, valve=None,
                 opt: TwoZoneOptions = TwoZoneOptions()):
        self.g, self.pp, self.valve, self.opt = geom, props, valve, opt
        s = props.saturated_state(T0)
        V_l = fill_frac * geom.V_total
        m_l, m_v = V_l * s["rho_l"], (geom.V_total - V_l) * s["rho_v"]
        self.m = m_l + m_v
        self.U = m_l * s["h_l"] + m_v * s["h_v"] - s["P"] * geom.V_total
        self.m0 = self.m
        self._flash()
        A_wet = geom.wetted_area(self.level())
        self.wet = WallRegion(wall_factory(T0), A_wet)
        self.dry = WallRegion(wall_factory(T0), geom.A_total - A_wet)
        self.t = self.m_out = self.H_out = self.Q_outer = 0.0
        self.U0, self.E_wall0 = self.U, self.E_wall()
        self.rows = []
        self.record({})

    def _flash(self):
        r = self.pp.uv(self.m / self.g.V_total, self.U / self.m)
        self.P, self.T, self.x = r["P"], r["T"], r["x"]
        s = self.pp.sat(self.P) if self.P < self.pp.P_c else None
        self.V_l = self.m * (1 - self.x) / s["rho_l"] if (s and 0 < self.x < 1) else \
            (self.g.V_total if self.x <= 0 else 0.0)

    def level(self):
        return self.g.level(self.V_l)

    def E_wall(self):
        return sum(r.col.energy() * r.scale for r in (self.wet, self.dry))

    def step(self):
        dt, g, pp, opt = self.opt.dt, self.g, self.pp, self.opt
        sat = pp.sat(self.P)
        hlev = self.level()
        Tw = self.wet.col.T_inner
        lp = pp.transport(self.P, 0.5 * (Tw + self.T), "liquid")
        q0, dq, regime = boiling_flux_and_derivative(Tw, self.T, sat, lp, g.liquid_char_length(hlev), opt.boiling)
        h_wet, Tref_wet = linearised_bc(q0, dq, Tw)
        Twd = self.dry.col.T_inner
        gp = pp.transport(self.P, 0.5 * (Twd + self.T), "gas")
        h_dry = h_free(gp, Twd - self.T, g.gas_char_length(hlev), opt.gas_wall_config)
        t_mid = self.t + 0.5 * dt
        Q = 0.0
        for reg, h_in, Tref in ((self.wet, h_wet, Tref_wet), (self.dry, h_dry, self.T)):
            col = reg.col
            Ts = col.T_outer
            qo, dqo, _, _ = col.outer(Ts, t_mid)
            col.step(dt, t_mid, Tref, h_in)
            self.Q_outer += (qo + dqo * (col.T_outer - Ts)) * col.A_out * reg.scale * dt
            Q += col.q_in * col.A_in * reg.scale
        mdot = 0.0
        if self.valve is not None and self.t >= self.valve.open_time and \
                (self.valve.set_pressure is None or self.P >= self.valve.set_pressure):
            zn = g.H if self.valve.z_nozzle is None else self.valve.z_nozzle
            if hlev >= zn:
                mdot = self.valve.liquid_mdot(self.P, sat["rho_l"])
                h_out = sat["h_l"]
            else:
                mdot = self.valve.gas_mdot(self.P, sat["rho_v"], pp.gas_gamma(self.P, self.T))
                h_out = sat["h_v"] if 0 < self.x < 1 else (self.U + self.P * g.V_total) / self.m
            mdot = min(mdot, 0.5 * self.m / dt)
            self.m -= mdot * dt
            self.U -= mdot * dt * h_out
            self.H_out += mdot * dt * h_out
            self.m_out += mdot * dt
        self.U += Q * dt
        self._flash()
        A_wet_new = g.wetted_area(self.level())
        dA = A_wet_new - self.wet.A
        if dA > 0:
            transfer_wall_area(self.dry, self.wet, dA)
        elif dA < 0:
            transfer_wall_area(self.wet, self.dry, -dA)
        self.t += dt
        self.record(dict(regime=regime, Q=Q, mdot=mdot))

    def record(self, info):
        E = self.U + self.E_wall() + self.H_out - self.Q_outer
        row = dict(t=self.t, P=self.P, T=self.T, x=self.x, level=self.level(),
                   Tw_wet=self.wet.col.T_inner, Tw_dry=self.dry.col.T_inner, m_out=self.m_out,
                   mass_err=(self.m + self.m_out - self.m0) / self.m0,
                   energy_err=(E - self.U0 - self.E_wall0) / max(abs(self.Q_outer) + abs(self.H_out), 1.0))
        row.update(info)
        self.rows.append(row)

    def run(self, t_end):
        while self.t < t_end - 1e-9:
            self.step()
        import pandas as pd
        return pd.DataFrame(self.rows)
