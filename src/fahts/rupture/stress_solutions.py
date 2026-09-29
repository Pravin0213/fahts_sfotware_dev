"""Stress solutions for a pressurised, heated cylindrical shell (closed form) and
equivalent stresses. See ``fahts.rupture`` for the references.
"""

from __future__ import annotations

import math

import numpy as np

from fahts.common.constants import P_ATM
from fahts.materials.en1993_mechanical import NU


def von_mises(sr, st, sz):
    return np.sqrt(0.5 * ((sr - st) ** 2 + (st - sz) ** 2 + (sz - sr) ** 2))


def tresca(sr, st, sz):
    s = np.stack([sr, st, sz])
    return s.max(0) - s.min(0)


# ------------------------------------------------------------------
# closed-form solutions
# ------------------------------------------------------------------


def membrane(p, a, t, s_ext):
    rm = a + 0.5 * t
    return -0.5 * p, p * rm / t, p * rm / (2 * t) + s_ext


def lame(p, a, b, r, s_ext):
    """Thick-wall pressure stresses (sr, st, sz) at radius r; closed ends."""
    k = p * a**2 / (b**2 - a**2)
    sz = k + s_ext
    return k * (1 - b**2 / r**2), k * (1 + b**2 / r**2), sz + 0 * r


def thermal_tg(r_nodes, e_nodes, r_eval, E, nu=NU):
    """Timoshenko & Goodier sec. 150, free ends, constant E: thermal stresses
    from thermal-strain profile e(r) (piecewise linear through r_nodes)."""
    a, b = r_nodes[0], r_nodes[-1]
    rr = np.linspace(a, b, 2001)
    ee = np.interp(rr, r_nodes, e_nodes)
    I = np.concatenate(
        ([0.0], np.cumsum(0.5 * (ee[1:] * rr[1:] + ee[:-1] * rr[:-1]) * np.diff(rr)))
    )
    Ib = I[-1]
    r = np.asarray(r_eval, float)
    Ir, er = np.interp(r, rr, I), np.interp(r, rr, ee)
    c = E / (1 - nu)
    sr = c / r**2 * ((r**2 - a**2) / (b**2 - a**2) * Ib - Ir)
    st = c / r**2 * ((r**2 + a**2) / (b**2 - a**2) * Ib + Ir - er * r**2)
    sz = c * (2 * Ib / (b**2 - a**2) - er)
    return sr, st, sz


def membrane_stresses(P, D, t, ext_long_mpa):
    """Thin-shell membrane stresses from absolute pressure P [Pa], diameter D,
    thickness t [m] and external longitudinal stress [MPa] -> (hoop, long,
    von Mises, Tresca) [MPa]."""
    Pg = (P - P_ATM) / 1e6
    rm = 0.5 * D + 0.5 * t
    s_h = Pg * rm / t
    s_l = Pg * rm / (2 * t) + ext_long_mpa
    s_r = -0.5 * Pg
    vm = math.sqrt(0.5 * ((s_h - s_l) ** 2 + (s_l - s_r) ** 2 + (s_r - s_h) ** 2))
    tr = max(s_h, s_l, s_r) - min(s_h, s_l, s_r)
    return s_h, s_l, vm, tr
