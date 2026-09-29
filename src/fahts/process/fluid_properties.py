"""Property dicts for the heat-transfer correlations from PR phases (mass basis, SI),
saturation property sets of a liquid pool, and pure-water properties (IAPWS via
CoolProp) for free-water phases.
"""

from __future__ import annotations

import CoolProp.CoolProp as CP
import numpy as np

from fahts.thermo import PRMixture


_W = CP.AbstractState("HEOS", "Water")


def phase_beta(p):
    """Isobaric expansion coefficient from the EOS derivatives of a Phase."""
    return -p.dPdT_v / (p.v * p.dPdv_T)


def phase_dict(m: PRMixture, p) -> dict:
    """rho, cp, mu, k, beta (SI, mass basis) for one Phase."""
    if p.name == "aqueous" or (m.iw is not None and p.x[m.iw] > 0.99):
        try:
            _W.update(CP.PT_INPUTS, p.P, p.T)
        except ValueError:                           # exactly saturated: liquid side
            _W.update(CP.PQ_INPUTS, p.P, 0.0 if p.name == "aqueous" else 1.0)
        return dict(rho=_W.rhomass(), cp=_W.cpmass(), mu=_W.viscosity(), k=_W.conductivity(),
                    beta=_W.isobaric_expansion_coefficient())
    rho_t = m.transport_rho(p)
    return dict(rho=p.rho, cp=p.cp_mass, mu=m.viscosity(p.x, p.T, rho_t),
                k=m.thermal_conductivity(p.x, p.T, rho_t), beta=phase_beta(p))


def film_gas(m: PRMixture, y, T, P) -> dict:
    return phase_dict(m, m.phase_props(y, T, P, "V"))


def sat_props(m: PRMixture, x, P, T_guess=None, water=False) -> dict:
    """Saturation property dict for twophase_physics correlations (bubble point of
    liquid composition x at P; pure-water properties for a free-water pool)."""
    if water:
        _W.update(CP.PQ_INPUTS, P, 0.0)
        s = dict(P=P, P_c=_W.p_critical(), M=18.015, T_sat=_W.T(), rho_l=_W.rhomass(),
                 h_l=_W.hmass(), mu_l=_W.viscosity(), k_l=_W.conductivity(), cp_l=_W.cpmass(),
                 sigma=_W.surface_tension())
        _W.update(CP.PQ_INPUTS, P, 1.0)
        s.update(rho_v=_W.rhomass(), h_v=_W.hmass(), mu_v=_W.viscosity(), k_v=_W.conductivity(),
                 cp_v=_W.cpmass())
        s["h_fg"] = s["h_v"] - s["h_l"]
        return s
    Tb, y = m.bubble_T(P, x, T_guess)
    L = m.phase_props(x, Tb, P, "L")
    V = m.phase_props(y, Tb, P, "V")
    dl, dv = phase_dict(m, L), phase_dict(m, V)
    sig = m.surface_tension(x, m.transport_rho(L), y, m.transport_rho(V))
    x = np.asarray(x) / np.sum(x)
    # Kay's-rule pseudo-critical pressure underestimates a mixture's true critical
    # pressure (often below the operating P for rich liquids), which would make the
    # reduced-pressure boiling correlations invalid (log of p_r > 1); cap p_r at 0.9,
    # the upper end of Cooper's (1984) data range.
    return dict(P=P, P_c=max(float(x @ m.Pc), P / 0.9), M=L.M * 1e3, T_sat=Tb,
                rho_l=L.rho, h_l=L.h_mass, mu_l=dl["mu"], k_l=dl["k"], cp_l=dl["cp"],
                rho_v=V.rho, h_v=V.h_mass, mu_v=dv["mu"], k_v=dv["k"], cp_v=dv["cp"],
                sigma=max(sig, 1e-5), h_fg=max(V.h_mass - L.h_mass, 1e3), y=y)
