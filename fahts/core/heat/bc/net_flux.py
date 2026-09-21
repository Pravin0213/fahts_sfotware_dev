"""
Phase 2.6 — EN 1993-1-2 §3.1 net heat flux to an exposed steel surface.

Sign convention (consistent throughout all heat-source and boundary-condition paths):
    **positive flux = heat flowing INTO the steel element**

This matches the FEM load vector Q in M·Ṫ + K·T = Q (SINTEF FAHTS Eq. 3.2.14/3.2.18),
where Q represents heat input to the element.  It is the *negation* of the surface-energy
notation used in FAHTS Theory Manual §3.2.4–§3.2.5, where E_s and E_c are defined as
positive when heat leaves the steel surface:

    Manual (positive = OUT of steel):  E_s = σ·ε·(T_s⁴ − T_fire⁴),  E_c = C·(T_s − T_g)
    Code   (positive = INTO steel):    q_rad = ε_m·σ·(T_fire_K⁴ − T_steel_K⁴)
                                       q_conv = h_conv·(T_fire − T_steel)

Consequences for each BC path:
  - FireZone convection/radiation: both positive when T_fire > T_steel (heating), negative
    when steel is hotter than fire (impossible under normal fire conditions, but physically
    consistent with the sign convention — the solver would show cooling, which is correct).
  - Steel re-radiation (prescribed-flux mode): q_rerad = −ε_steel·σ·T_steel_K⁴ < 0 (heat
    OUT of steel) — always negative, applied as a negative contribution to Q in surface_solver.
  - ConcentratedSource / LineSource: q = E·cos(θ)/(4π·r²) ≥ 0 — always positive (heat into
    steel), clamped to 0 for back-facing quads (no back-radiation from point sources).
  - RadiationBall: flux is a positive prescribed irradiance (heat into steel); steel
    re-radiation is handled by the separate eps_rerad path in SurfaceTransientSolver.
  - View factors (§3.3.4): F·ε_fire·T_fire_K⁴ absorbed minus T_steel_K⁴ re-emitted — net
    positive when fire temperature dominates (heat in), negative when steel is hotter.

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
