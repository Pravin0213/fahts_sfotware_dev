"""Nusselt film condensation on the inner wall (vertical wall / horizontal cylinder).
"""

from __future__ import annotations

from fahts.common.constants import G_STANDARD


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
        return C * (G_STANDARD * rl * (rl - rv) * sat["k_l"]**3 * hfg / (sat["mu_l"] * L * d))**0.25

    h = h_of(dT)
    q = -h * dT
    e = 1e-3 * dT
    dq = (-(h_of(dT - e) * (dT - e)) + h_of(dT + e) * (dT + e)) / (2 * e)  # d(-q)/d(dT)=d q/dT_w
    return q, dq, h
