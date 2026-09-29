"""Pool boiling correlations: nucleate boiling (Rohsenow, Mostinski, Cooper, Gorenflo),
critical heat flux, minimum film boiling flux and film boiling.
"""

from __future__ import annotations

import math

from fahts.common.constants import G_STANDARD, SIGMA


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
    A = sat["mu_l"] * sat["h_fg"] * math.sqrt(G_STANDARD * (sat["rho_l"] - sat["rho_v"]) / sat["sigma"])
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
    return math.sqrt(sat["sigma"] / (G_STANDARD * (sat["rho_l"] - sat["rho_v"])))


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
    base = sat["h_fg"] * math.sqrt(sat["rho_v"]) * (sat["sigma"] * G_STANDARD * (sat["rho_l"] - sat["rho_v"]))**0.25
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
    return 0.09 * rv * sat["h_fg"] * (sat["sigma"] * G_STANDARD * (rl - rv) / (rl + rv)**2)**0.25


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
    h_fb = C * (sat["k_v"]**3 * rv * G_STANDARD * (rl - rv) * hfg / (sat["mu_v"] * dT * L))**0.25
    Ts = T_w - dT
    h_rad = eps_w * SIGMA * (T_w**4 - Ts**4) / dT
    return h_fb + 0.75 * h_rad
