"""Wet-wall heat flux on the nucleate-boiling branch only (no CHF cap).
"""

from __future__ import annotations

from fahts.process.inner_ht.boiling_correlations import nb_cooper
from fahts.process.inner_ht.natural_convection import q_free


def nucleate_only_flux(T_w, T_l, sat, liq_props, L_nc, n_blend=3.0):
    """Wall -> pool heat flux on the nucleate-boiling branch only (Cooper 1984, blended
    with free convection, Churchill-Usagi), no critical-heat-flux limit. Cooper needs
    only reduced pressure, molar mass and q, so it stays defined near the critical
    point where h_fg and sigma -> 0 (and CHF would collapse)."""
    q_nc = q_free(T_w, T_l, liq_props, L_nc, "vertical_plate")
    dT = T_w - sat["T_sat"]
    if dT <= 0.0:
        return q_nc
    q_nb = nb_cooper(sat)(dT)
    return (max(q_nc, 0.0) ** n_blend + q_nb ** n_blend) ** (1.0 / n_blend)
