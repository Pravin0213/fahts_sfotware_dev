"""Full pool boiling curve q(T_w) (natural convection -> nucleate -> CHF -> transition
-> film) with dq/dT_w for implicit wall coupling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from scipy.optimize import brentq

from fahts.process.inner_ht.boiling_correlations import (
    h_film_boiling,
    nb_cooper,
    nb_gorenflo,
    nb_mostinski,
    nb_rohsenow,
    q_chf,
    q_min_film,
)
from fahts.process.inner_ht.natural_convection import q_free


# ------------------------------------------------------------------ boiling curve
@dataclass
class BoilingModel:
    """Settings for the wall->liquid boiling curve."""

    nucleate: str = "cooper"  # rohsenow | mostinski | cooper | gorenflo (see notes)
    chf: str = "zuber"  # zuber | large_plate | cylinder
    film: str = "berenson"  # berenson | bromley
    nc_config: str = "vertical_plate"  # liquid free convection correlation
    blend_n: float = 3.0  # Churchill-Usagi power for nc + nucleate
    dT_onb: float = 0.0  # onset-of-boiling wall superheat [K]
    subcooled_chf: bool = True
    C_sf: float = 0.013  # Rohsenow
    Rp_um: float = 1.0  # Cooper
    h0_gorenflo: float | None = None  # Gorenflo reference h0
    water: bool = False
    eps_w: float = 0.8  # wall emissivity (film boiling radiation)
    D: float | None = None  # heater diameter (bromley / cylinder CHF)

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


def boiling_flux(
    T_w: float,
    T_l: float,
    sat: dict,
    liq_props: dict,
    L_nc: float,
    model: BoilingModel = BoilingModel(),
) -> tuple[float, str]:
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
        q = (max(q_nc, 0.0) ** n + q_nb**n) ** (1 / n)
        return min(q, qmax) if q_nb < qmax else qmax, "nucleate"
    qmin = q_min_film(sat)
    f_film = (
        lambda d: h_film_boiling(sat, d, sat["T_sat"] + d, model.film, model.D, model.eps_w) * d
    )
    dT_min = _solve_monotone(lambda d: f_film(d) - qmin, 1e-2, 2000.0)
    if dT_min <= dT_chf * 1.05:  # near-critical: no distinct transition branch
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
