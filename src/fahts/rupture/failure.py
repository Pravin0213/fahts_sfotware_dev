"""Stress / allowable time series over a model output and failure (rupture) times per
stress variant, allowable basis and criterion. See ``fahts.rupture``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fahts.common.compat import trapezoid
from fahts.common.constants import P_ATM
from fahts.materials import SteelTable
from fahts.materials.en1993_mechanical import E_of_T, eps_th
from fahts.rupture.gps_fe import GeneralizedPlaneStrainFE
from fahts.rupture.stress_solutions import lame, membrane, thermal_tg, tresca, von_mises


def crossing(time, s, allow):
    d = np.asarray(s) - np.asarray(allow)
    i = np.flatnonzero(d >= 0)
    if len(i) == 0:
        return None
    i = i[0]
    if i == 0:
        return float(time[0])
    return float(time[i - 1] + (time[i] - time[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1]))


def columns(py):
    return [c[: -len("_T1_C")] for c in py.columns if c.endswith("_T1_C")]


def stress_series(
    py: pd.DataFrame,
    seg: dict,
    mat: SteelTable,
    col: str = "background",
    *,
    x_nodes,
    n_el: int = 120,
) -> pd.DataFrame:
    """Stress/allowable time series for one wall column.

    seg: case['seg'] (D, t, strength_mpa, stress_factor, stress_type, ext_long_mpa).
    x_nodes: wall node positions from the inner surface [m] as used by the wall model
    (scaled to the wall thickness t here).
    Returns a DataFrame with, per location L in (in, mid, out) and per variant,
    sigma_vM, sigma_Tr, and allowables."""
    a = 0.5 * seg["D"]
    t = seg["t"]
    b = a + t
    s_ext = seg.get("ext_long_mpa", 0.0)
    F = mat.f_uts_at if seg.get("stress_type", "U") == "U" else mat.f_yield_at
    sf = (
        lambda T_C: seg["strength_mpa"]
        * seg.get("stress_factor", 1.0)
        * F(np.asarray(T_C) + 273.15)
    )
    r_T = a + np.asarray(x_nodes, float) * (t / x_nodes[-1])
    Tcols = [f"{col}_T{i+1}_C" for i in range(len(x_nodes))]
    TT = py[Tcols].to_numpy()
    T_ref = float(TT[0].mean())
    locs = {"in": a, "mid": 0.5 * (a + b), "out": b}
    r_loc = np.array(list(locs.values()))
    gps = GeneralizedPlaneStrainFE(a, b, n_el)
    rr = np.linspace(a, b, 401)

    rows = []
    for k in range(len(py)):
        p = max(py.P_bara.iat[k] * 1e5 - P_ATM, 0.0) / 1e6
        T = TT[k]
        Tm = (
            float(py[f"{col}_T_mean_C"].iat[k])
            if f"{col}_T_mean_C" in py
            else float(trapezoid(T * r_T, r_T) / trapezoid(r_T, r_T))
        )
        T_loc = np.interp(r_loc, r_T, T)
        r = dict(
            Time=py.Time.iat[k],
            p_MPa=p,
            T_mean_C=Tm,
            allow_mean=float(sf(Tm)),
            allow_limit=float(trapezoid(sf(np.interp(rr, r_T, T)), rr) / t),
        )
        for L, TL in zip(locs, T_loc):
            r[f"T_{L}_C"] = TL
            r[f"allow_local_{L}"] = float(sf(TL))
        # membrane
        sr, st, sz = membrane(p, a, t, s_ext)
        r["mem_vM"], r["mem_Tr"] = float(von_mises(sr, st, sz)), float(tresca(sr, st, sz))
        # Lame
        Fz = p * math.pi * a**2 + s_ext * math.pi * (b**2 - a**2)
        lr, lt, lz = lame(p, a, b, r_loc, s_ext)
        # FE: pressure + thermal (E(T))
        sfun = gps.solve_vec(r_T, T, T_ref, p, Fz)
        fr, ft, fz = sfun(r_loc)
        # closed form T&G thermal with E at T_mean, + Lame
        e_nodes = eps_th(T) - eps_th(T_ref)
        tr_, tt_, tz_ = thermal_tg(r_T, e_nodes, r_loc, float(E_of_T(Tm)))
        for i, L in enumerate(locs):
            r[f"lame_vM_{L}"] = float(von_mises(lr[i], lt[i], lz[i]))
            r[f"lame_Tr_{L}"] = float(tresca(lr[i], lt[i], lz[i]))
            r[f"th_vM_{L}"] = float(von_mises(fr[i], ft[i], fz[i]))
            r[f"th_Tr_{L}"] = float(tresca(fr[i], ft[i], fz[i]))
            r[f"th_hoop_{L}"], r[f"th_axial_{L}"], r[f"th_radial_{L}"] = (
                float(ft[i]),
                float(fz[i]),
                float(fr[i]),
            )
            r[f"tg_vM_{L}"] = float(von_mises(lr[i] + tr_[i], lt[i] + tt_[i], lz[i] + tz_[i]))
            r[f"tg_Tr_{L}"] = float(tresca(lr[i] + tr_[i], lt[i] + tt_[i], lz[i] + tz_[i]))
        rows.append(r)
    return pd.DataFrame(rows)


VARIANTS = ["membrane", "lame", "thermal_elastic", "thermal_tg", "thermal_2x", "limit"]


def failure_times(ss: pd.DataFrame) -> list[dict]:
    """Failure times per variant x allowable basis x criterion from stress_series()."""
    tm = ss.Time.to_numpy()
    out = []

    def add(variant, basis, crit, pairs):
        best, where = None, None
        for L, s, al in pairs:
            tf = crossing(tm, s, al)
            if tf is not None and (best is None or tf < best):
                best, where = tf, L
        out.append(
            dict(variant=variant, allow_basis=basis, criterion=crit, t_fail_s=best, location=where)
        )

    for crit, c in (("vonMises", "vM"), ("Tresca", "Tr")):
        add("membrane", "mean", crit, [("mem", ss[f"mem_{c}"], ss.allow_mean)])
        add("membrane", "local", crit, [("mem", ss[f"mem_{c}"], ss.allow_local_out)])
        add("limit", "through-wall avg", crit, [("mem", ss[f"mem_{c}"], ss.allow_limit)])
        for v, pre, fac in (
            ("lame", "lame", 1),
            ("thermal_elastic", "th", 1),
            ("thermal_tg", "tg", 1),
            ("thermal_2x", "th", 2),
        ):
            for basis in ("local", "mean"):
                pairs = [
                    (
                        L,
                        ss[f"{pre}_{c}_{L}"],
                        fac * (ss[f"allow_local_{L}"] if basis == "local" else ss.allow_mean),
                    )
                    for L in ("in", "mid", "out")
                ]
                add(v, basis, crit, pairs)
    return out


def evaluate(py: pd.DataFrame, case: dict, mat: SteelTable, col: str | None = None, **kw):
    """Run stress_series on the hottest wall column (max final T_mean) and
    return (stress series DataFrame, failure-time DataFrame)."""
    seg = case["seg"]
    if col is None:
        cs = columns(py)
        col = (
            max(
                cs,
                key=lambda c: (
                    py[f"{c}_T_mean_C"].max(skipna=True)
                    if py[f"{c}_T_mean_C"].notna().any()
                    else -np.inf
                ),
            )
            if cs
            else "background"
        )
    ss = stress_series(py, seg, mat, col, **kw)
    return ss, pd.DataFrame(failure_times(ss))
