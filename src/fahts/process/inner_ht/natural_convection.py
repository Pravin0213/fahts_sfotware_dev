"""Natural convection from the inner wall to a single-phase fluid (free-convection
Nusselt correlations by configuration), and the named gas-side correlations used by
the vessel model (``H_CORRELATIONS``).
"""

from __future__ import annotations

from fahts.common.constants import G_STANDARD


def rayleigh(props: dict, dT: float, L: float) -> tuple[float, float]:
    """Ra = g beta |dT| L^3 / (nu alpha), Pr. props at film temperature."""
    Pr = props["cp"] * props["mu"] / props["k"]
    Ra = (G_STANDARD * abs(props["beta"]) * abs(dT) * L**3 * props["rho"]**2 * props["cp"]
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
