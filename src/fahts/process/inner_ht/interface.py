"""Heat and mass exchange across the gas-liquid free surface: horizontal-surface free
convection on both sides and the interface energy balance.
"""

from __future__ import annotations

from fahts.process.inner_ht.natural_convection import h_free


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
