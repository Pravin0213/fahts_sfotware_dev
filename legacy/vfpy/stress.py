"""
stress.py - through-wall pressure + thermal stress in the cylindrical shell and
failure-time evaluation for several stress/strength variants.

Written only from published mechanics and material data (no VessFire internals).

Geometry and loads
------------------
Long hollow cylinder, inner radius a = D/2, outer radius b = a + t, gauge
internal pressure p (outside = atmosphere), closed ends, plus an externally
applied longitudinal membrane stress sigma_ext (case field ext_long_mpa) acting
on the shell cross-section.  Axial resultant per the free-body of a closed
vessel:  F_z = p*pi*a^2 + sigma_ext*pi*(b^2 - a^2).

Radial temperature T(r) from the model's wall nodes (column {col}_T1_C ...
{col}_T12_C at r_i = a + x_i, x_i = heat_transfer.VESSFIRE_LOG_NODES_105MM),
linearly interpolated in r.  Stress-free reference state = initial shell
temperature (first output row).

Stress solutions
----------------
1. Membrane (thin shell, as vessel.stresses): hoop = p*rm/t,
   long = p*rm/(2t) + sigma_ext, radial = -p/2, rm = a + t/2.

2. Lame thick-wall pressure solution (Timoshenko & Goodier, Theory of
   Elasticity, 3rd ed. 1970, sec. 28 "Thick-walled cylinder"):
       sigma_r = p a^2/(b^2-a^2) (1 - b^2/r^2)
       sigma_t = p a^2/(b^2-a^2) (1 + b^2/r^2)
       sigma_z = F_z / (pi (b^2-a^2))

3. Thermal stress, generalized plane strain (uniform axial strain eps_z,
   zero thermal axial resultant = "free ends" far from the ends), Timoshenko &
   Goodier sec. 150 (long hollow circular cylinder, eqs. 244-246 in 3rd ed.):
       sigma_r = E/((1-nu) r^2) [ (r^2-a^2)/(b^2-a^2) I(b) - I(r) ]
       sigma_t = E/((1-nu) r^2) [ (r^2+a^2)/(b^2-a^2) I(b) + I(r) - e(r) r^2 ]
       sigma_z = E/(1-nu) [ 2 I(b)/(b^2-a^2) - e(r) ]
   with e(r) = thermal strain (alpha*T for constant alpha) and
   I(r) = int_a^r e(s) s ds.  Valid for constant E, nu: implemented in
   thermal_tg() with E at the through-wall mean temperature (check solution).

   Because E(T) drops by >50 % between 400 and 700 C, the main solution is a
   1-D axisymmetric generalized-plane-strain finite-element model (linear
   elements, 2-point Gauss) with E(r) = E(T(r)), thermal strain eps_th(T(r)),
   internal pressure on r = a and axial force F_z conjugate to eps_z
   (class GPS).  With constant E it reproduces 2.+3. (see _selftest()).

Material data (carbon steel)
----------------------------
EN 1993-1-2:2005 (Eurocode 3, structural fire design):
  * E(T) = 210 GPa * k_E,theta, Table 3.1 (reduction factor for the slope of
    the linear elastic range).
  * thermal elongation Delta l / l, clause 3.4.1.1 eq. (3.1a-c):
      20 <= T < 750 C : 1.2e-5 T + 0.4e-8 T^2 - 2.416e-4
      750 <= T <= 860 : 1.1e-2
      860 < T <= 1200 : 2e-5 T - 6.2e-3
  * nu = 0.3 (EN 1993-1-1 cl. 3.2.6; taken temperature independent).

Strength (shared case input, not VessFire internals): sigma_f(T) =
strength_mpa * stress_factor * F(T), F = F_UTS (stress type U) or F_Yield,
table from vessfire.db through heat_transfer.Material.

Equivalent stresses: von Mises  sqrt(0.5*((s1-s2)^2+(s2-s3)^2+(s3-s1)^2)),
Tresca  max(s) - min(s)  (both compared with sigma_f).

Variants (see evaluate()):
  membrane        membrane stresses; allowable at T_mean ("mean") and at the
                  hottest (outer) surface temperature ("local").  "mean" is the
                  current vessel.py check.
  lame            Lame pressure stresses only, at inner / mid / outer surface.
  thermal_elastic Lame + elastic thermal stress (FE, E(T)), inner/mid/outer.
  thermal_tg      as thermal_elastic but closed-form T&G with E at T_mean.
  thermal_2x      thermal_elastic P+Q equivalent vs 2*sigma_f: the classic
                  primary+secondary (shakedown) limit, ASME VIII-2 5.5.6
                  (S_PS) / EN 13445-3 Annex C C.7.  A code design check, not a
                  rupture criterion; reported for orientation only.
  limit           membrane stresses vs the through-thickness average flow
                  stress (1/t) int_a^b sigma_f(T(r)) dr.  Justification:
                  thermal stress is self-equilibrating (secondary, ASME
                  VIII-2 5.2.2 / EN 13445-3 Annex C) and does not change the
                  plastic collapse load of a ductile elastic-perfectly-plastic
                  structure (limit theorems; e.g. Lubliner, Plasticity Theory,
                  1990, sec. 4.4/5.2; EN 13445-3 Annex B, B.8.2 GPD check uses
                  mechanical actions only), while the membrane (collapse)
                  capacity of a section with through-thickness varying flow
                  stress is N0 = int sigma_f dz (Ilyushin's generalized
                  stress; Hodge, Plastic Analysis of Structures, 1959).  For
                  Tresca this is Hill's thick-cylinder limit pressure
                  p_L = int sigma_f/r dr (Hill, Math. Theory of Plasticity,
                  1950, ch. V) to within (b-a)/a ~ 1 %.

Allowable basis for the through-wall variants: "local" = sigma_f(T at the
evaluation radius), "mean" = sigma_f(T_mean).  Failure time = first time the
equivalent stress reaches the allowable at any of the evaluation points
(linear interpolation between output rows).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from heat_transfer import Material, VESSFIRE_LOG_NODES_105MM

P_ATM = 101325.0

# EN 1993-1-2 Table 3.1, carbon steel: k_E,theta
_EC3_T = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200.0])
_EC3_KE = np.array([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.31, 0.13, 0.09, 0.0675, 0.045, 0.0225, 0.0])
E20 = 210e3      # MPa, EN 1993-1-1 3.2.6
NU = 0.3         # EN 1993-1-1 3.2.6
E_FLOOR = 1e-3   # keep FE stiffness non-singular at >= 1200 C


def E_of_T(T_C):
    """Young's modulus [MPa], EN 1993-1-2 Table 3.1."""
    return E20 * np.maximum(np.interp(T_C, _EC3_T, _EC3_KE), E_FLOOR)


def eps_th(T_C):
    """Thermal elongation Delta l/l relative to 20 C, EN 1993-1-2 3.4.1.1."""
    T = np.asarray(T_C, float)
    return np.where(T < 750, 1.2e-5 * T + 0.4e-8 * T**2 - 2.416e-4,
                    np.where(T <= 860, 1.1e-2, 2e-5 * T - 6.2e-3))


# ------------------------------------------------------------------
# equivalent stresses
# ------------------------------------------------------------------

def von_mises(sr, st, sz):
    return np.sqrt(0.5 * ((sr - st)**2 + (st - sz)**2 + (sz - sr)**2))


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
    I = np.concatenate(([0.0], np.cumsum(0.5 * (ee[1:] * rr[1:] + ee[:-1] * rr[:-1]) * np.diff(rr))))
    Ib = I[-1]
    r = np.asarray(r_eval, float)
    Ir, er = np.interp(r, rr, I), np.interp(r, rr, ee)
    c = E / (1 - nu)
    sr = c / r**2 * ((r**2 - a**2) / (b**2 - a**2) * Ib - Ir)
    st = c / r**2 * ((r**2 + a**2) / (b**2 - a**2) * Ib + Ir - er * r**2)
    sz = c * (2 * Ib / (b**2 - a**2) - er)
    return sr, st, sz


# ------------------------------------------------------------------
# 1-D generalized plane strain FE (variable E, thermal strain)
# ------------------------------------------------------------------

class GPS:
    """Axisymmetric generalized-plane-strain FE of a hollow cylinder.
    Unknowns: nodal radial displacements u_i and the uniform axial strain eps_z.
    Units MPa, m (forces MN per m length)."""

    def __init__(self, a, b, n_el=120, nu=NU):
        s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_el + 1)))   # graded to both surfaces
        self.r = a + (b - a) * s
        self.a, self.b, self.nu, self.n = a, b, nu, n_el
        g = 1 / math.sqrt(3)
        r1, r2 = self.r[:-1], self.r[1:]
        self.h = r2 - r1
        self.rg = np.stack([0.5 * (r1 + r2) - 0.5 * g * self.h, 0.5 * (r1 + r2) + 0.5 * g * self.h])  # (2, n)
        nu_ = nu
        self.D1 = np.array([[1 - nu_, nu_, nu_], [nu_, 1 - nu_, nu_], [nu_, nu_, 1 - nu_]]) / ((1 + nu_) * (1 - 2 * nu_))

    def _B(self, e, r):
        r1, h = self.r[e], self.h[e]
        N1, N2 = (self.r[e + 1] - r) / h, (r - r1) / h
        return np.array([[-1 / h, 1 / h, 0.0], [N1 / r, N2 / r, 0.0], [0.0, 0.0, 1.0]])

    def solve(self, E_fn, e_fn, p, F_z):
        """E_fn(r)->E [MPa], e_fn(r)->thermal strain. Returns function
        stress(r_eval) -> (sr, st, sz)."""
        n = self.n + 1
        K = np.zeros((n + 1, n + 1))
        f = np.zeros(n + 1)
        for e in range(self.n):
            idx = [e, e + 1, n]
            for q in range(2):
                r = self.rg[q, e]
                B = self._B(e, r)
                D = E_fn(r) * self.D1
                w = 0.5 * self.h[e] * 2 * np.pi * r
                K[np.ix_(idx, idx)] += B.T @ D @ B * w
                f[idx] += B.T @ D @ (e_fn(r) * np.ones(3)) * w
        f[0] += p * 2 * np.pi * self.a
        f[n] += F_z
        x = np.linalg.solve(K, f)
        u, ez = x[:n], x[n]

        # radial strain at the Gauss points (superconvergent), recovered to
        # arbitrary r by linear inter/extrapolation; hoop strain u/r and eps_z
        # are taken exactly.
        rg = self.rg.T.ravel()
        er_g = np.repeat((u[1:] - u[:-1]) / self.h, 2)

        def stress(r_eval):
            out = []
            for r in np.atleast_1d(r_eval):
                e = min(max(np.searchsorted(self.r, r) - 1, 0), self.n - 1)
                ui = u[e] + (u[e + 1] - u[e]) * (r - self.r[e]) / self.h[e]
                if r <= rg[0] or r >= rg[-1]:
                    j = 0 if r <= rg[0] else len(rg) - 2
                    # extrapolate using first/last two elements' Gauss values
                    j2 = [0, 2] if j == 0 else [len(rg) - 3, len(rg) - 1]
                    x0, x1 = rg[j2]
                    y0, y1 = er_g[j2]
                    er = y0 + (y1 - y0) * (r - x0) / (x1 - x0)
                else:
                    er = np.interp(r, rg, er_g)
                eps = np.array([er, ui / r, ez]) - e_fn(r)
                out.append(E_fn(r) * self.D1 @ eps)
            return np.array(out).T
        return stress

    def solve_vec(self, r_T, T_C, T_ref_C, p, F_z):
        """Convenience: temperature nodes -> stress function (EN 1993-1-2 data)."""
        e_ref = float(eps_th(T_ref_C))
        E_fn = lambda r: E_of_T(np.interp(r, r_T, T_C))
        e_fn = lambda r: eps_th(np.interp(r, r_T, T_C)) - e_ref
        return self.solve(E_fn, e_fn, p, F_z)


# ------------------------------------------------------------------
# evaluation over a model output DataFrame
# ------------------------------------------------------------------

def _crossing(time, s, allow):
    d = np.asarray(s) - np.asarray(allow)
    i = np.flatnonzero(d >= 0)
    if len(i) == 0:
        return None
    i = i[0]
    if i == 0:
        return float(time[0])
    return float(time[i - 1] + (time[i] - time[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1]))


def columns(py):
    return [c[:-len("_T1_C")] for c in py.columns if c.endswith("_T1_C")]


def stress_series(py: pd.DataFrame, seg: dict, mat: Material, col: str = "background",
                  x_nodes=VESSFIRE_LOG_NODES_105MM, n_el: int = 120) -> pd.DataFrame:
    """Stress/allowable time series for one wall column.

    seg: case['seg'] (D, t, strength_mpa, stress_factor, stress_type, ext_long_mpa).
    Returns a DataFrame with, per location L in (in, mid, out) and per variant,
    sigma_vM, sigma_Tr, and allowables."""
    a = 0.5 * seg["D"]
    t = seg["t"]
    b = a + t
    s_ext = seg.get("ext_long_mpa", 0.0)
    F = mat.f_uts_at if seg.get("stress_type", "U") == "U" else mat.f_yield_at
    sf = lambda T_C: seg["strength_mpa"] * seg.get("stress_factor", 1.0) * F(np.asarray(T_C) + 273.15)
    r_T = a + np.asarray(x_nodes, float) * (t / x_nodes[-1])
    Tcols = [f"{col}_T{i+1}_C" for i in range(len(x_nodes))]
    TT = py[Tcols].to_numpy()
    T_ref = float(TT[0].mean())
    locs = {"in": a, "mid": 0.5 * (a + b), "out": b}
    r_loc = np.array(list(locs.values()))
    gps = GPS(a, b, n_el)
    rr = np.linspace(a, b, 401)

    rows = []
    for k in range(len(py)):
        p = max(py.P_bara.iat[k] * 1e5 - P_ATM, 0.0) / 1e6
        T = TT[k]
        Tm = float(py[f"{col}_T_mean_C"].iat[k]) if f"{col}_T_mean_C" in py else float(np.trapezoid(T * r_T, r_T) / np.trapezoid(r_T, r_T))
        T_loc = np.interp(r_loc, r_T, T)
        r = dict(Time=py.Time.iat[k], p_MPa=p, T_mean_C=Tm,
                 allow_mean=float(sf(Tm)),
                 allow_limit=float(np.trapezoid(sf(np.interp(rr, r_T, T)), rr) / t))
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
            r[f"th_hoop_{L}"], r[f"th_axial_{L}"], r[f"th_radial_{L}"] = float(ft[i]), float(fz[i]), float(fr[i])
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
            tf = _crossing(tm, s, al)
            if tf is not None and (best is None or tf < best):
                best, where = tf, L
        out.append(dict(variant=variant, allow_basis=basis, criterion=crit, t_fail_s=best, location=where))

    for crit, c in (("vonMises", "vM"), ("Tresca", "Tr")):
        add("membrane", "mean", crit, [("mem", ss[f"mem_{c}"], ss.allow_mean)])
        add("membrane", "local", crit, [("mem", ss[f"mem_{c}"], ss.allow_local_out)])
        add("limit", "through-wall avg", crit, [("mem", ss[f"mem_{c}"], ss.allow_limit)])
        for v, pre, fac in (("lame", "lame", 1), ("thermal_elastic", "th", 1), ("thermal_tg", "tg", 1),
                            ("thermal_2x", "th", 2)):
            for basis in ("local", "mean"):
                pairs = [(L, ss[f"{pre}_{c}_{L}"],
                          fac * (ss[f"allow_local_{L}"] if basis == "local" else ss.allow_mean))
                         for L in ("in", "mid", "out")]
                add(v, basis, crit, pairs)
    return out


def evaluate(py: pd.DataFrame, case: dict, col: str | None = None, **kw):
    """Run stress_series on the hottest wall column (max final T_mean) and
    return (stress series DataFrame, failure-time DataFrame)."""
    seg = case["seg"]
    mat = Material.from_vessfire_db(seg["material"])
    if col is None:
        cs = columns(py)
        col = max(cs, key=lambda c: py[f"{c}_T_mean_C"].max()) if cs else "background"
    ss = stress_series(py, seg, mat, col, **kw)
    return ss, pd.DataFrame(failure_times(ss))


def _selftest():
    """FE vs closed form (constant E): pressure (Lame) + linear-in-ln r thermal."""
    a, b, p = 1.29, 1.395, 20.0
    s_ext = 30.0
    Fz = p * math.pi * a**2 + s_ext * math.pi * (b**2 - a**2)
    rT = np.linspace(a, b, 12)
    e = 1.3e-5 * 300 * np.log(rT / a) / np.log(b / a)
    g = GPS(a, b, 200)
    f = g.solve(lambda r: 2e5, lambda r: np.interp(r, rT, e), p, Fz)
    rl = np.array([a, 0.5 * (a + b), b])
    fe = np.array(f(rl))
    lr, lt, lz = lame(p, a, b, rl, s_ext)
    tr_, tt_, tz_ = thermal_tg(rT, e, rl, 2e5)
    cf = np.array([lr + tr_, lt + tt_, lz + tz_])
    return np.abs(fe - cf).max(), cf


if __name__ == "__main__":
    err, cf = _selftest()
    print("self-test max |FE - closed form| [MPa]:", round(float(err), 3))
    print(np.round(cf, 1))
