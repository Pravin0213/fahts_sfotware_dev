"""
Phase 2.6 — EN 1993-1-2 §3.1 net heat flux to an exposed steel surface.

q_net = q_rad + q_conv
    q_rad  = ε_m · σ · (T_fire_K⁴ − T_steel_K⁴)   [W/m²]
    q_conv = h_conv · (T_fire − T_steel)              [W/m²]
"""
from __future__ import annotations

_SIGMA: float = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]


def net_heat_flux(
    T_fire: float,
    T_steel: float,
    epsilon_m: float,
    h_conv: float,
    sigma: float = _SIGMA,
) -> float:
    """
    Compute EN 1993-1-2 Eq. (3.1) net heat flux to a steel surface.

    Parameters
    ----------
    T_fire    : fire compartment (gas) temperature [°C]
    T_steel   : steel surface temperature [°C]
    epsilon_m : resultant emissivity = ε_fire × ε_steel  (dimensionless, 0–1)
    h_conv    : convective coefficient [W/(m²·K)]
                EN 1993-1-2 recommends 25 (standard) or 50 (hydrocarbon)
    sigma     : Stefan-Boltzmann constant [W/(m²·K⁴)]; default 5.67×10⁻⁸

    Returns
    -------
    float
        Net heat flux q_net [W/m²].  Positive = heat flowing *into* steel.
    """
    T_fire_K  = T_fire  + 273.15
    T_steel_K = T_steel + 273.15
    q_rad  = epsilon_m * sigma * (T_fire_K ** 4 - T_steel_K ** 4)
    q_conv = h_conv * (T_fire - T_steel)
    return q_rad + q_conv


def radiative_flux(
    T_fire: float,
    T_steel: float,
    epsilon_m: float,
    sigma: float = _SIGMA,
) -> float:
    """Radiative component only [W/m²]."""
    return epsilon_m * sigma * ((T_fire + 273.15) ** 4 - (T_steel + 273.15) ** 4)


def convective_flux(T_fire: float, T_steel: float, h_conv: float) -> float:
    """Convective component only [W/m²]."""
    return h_conv * (T_fire - T_steel)
