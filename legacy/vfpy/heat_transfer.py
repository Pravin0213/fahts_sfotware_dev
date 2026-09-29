"""
heat_transfer.py - 1D transient radial heat conduction through a fire-exposed
cylindrical vessel wall.

Built from published methods only (no VessFire internals):
  * Fire boundary: the specified heat flux is converted to an equivalent fire
    gas temperature T_f, which then drives radiation and convection:
      q_net(T_s) = eps_s*sigma*(eps_f*T_f^4 - T_s^4) + h_f*(T_f - T_s)
    T_f is solved so that q_net at a reference surface temperature equals the
    specified heat load. VessFire's published description states the flux is
    split "based on initial conditions of the exposed object", so the reference
    is the initial shell temperature. With eps_f = 1 this is the standard
    fire-gas-temperature boundary (EN 1991-1-2 form).
    Alternative (t_air_mode="blackbody"): the specified flux is taken as
    incident black-body radiation, sigma*T_f^4 = q_spec, and the absorbed flux
    follows from the balance above without being forced to equal q_spec.
  * Conduction: implicit (backward Euler) vertex-centred finite volume in
    cylindrical coordinates, tridiagonal system solved by the Thomas algorithm
    (same pattern as the LNG insulation solver this was adapted from).
  * Inner boundary: convection to the gas, h supplied by the caller.
  * Material: temperature-dependent cp and k, linear interpolation in a table.

One WallColumn = one radial line of nodes through the shell, representing an
area of the vessel that sees one heat load (e.g. background or peak zone).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

SIGMA = 5.670374419e-8

# Node positions through the wall, measured from the INNER surface [m], as
# printed by VessFire in its run log ("r [m]" row) for the 105 mm shell.
# Two surface nodes plus ten interior nodes.
VESSFIRE_LOG_NODES_105MM = np.array([
    0.0, 0.00525, 0.01575, 0.02625, 0.03675, 0.04725,
    0.05775, 0.06825, 0.07875, 0.08925, 0.1017, 0.105])


# ============================================================
# MATERIAL
# ============================================================

@dataclass
class Material:
    """Temperature-dependent steel properties (T in K)."""
    name: str
    T: np.ndarray
    cp: np.ndarray          # J/kg/K
    k: np.ndarray           # W/m/K
    rho: float              # kg/m3 (constant)
    f_yield: np.ndarray     # yield retention factor
    f_uts: np.ndarray       # UTS retention factor

    def cp_at(self, T):
        return np.interp(T, self.T, self.cp)

    def k_at(self, T):
        return np.interp(T, self.T, self.k)

    def f_yield_at(self, T):
        return np.interp(T, self.T, self.f_yield)

    def f_uts_at(self, T):
        return np.interp(T, self.T, self.f_uts)

    @classmethod
    def from_vessfire_db(cls, name: str,
                         db=Path(r"C:\Program Files\VessFire\VessFire14\vessfire.db")):
        """Read the material table the VessFire case uses, so both models get
        identical property input. Opened read-only."""
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "select temperature, Cp, Conduct, Dens, F_Yield, F_UTS from metals "
            "where name = ? order by temperature", (name,)).fetchall()
        con.close()
        if not rows:
            raise ValueError(f"material {name!r} not found in {db}")
        a = np.array([[r[0], r[1], r[2], r[4], r[5]] for r in rows], dtype=float)
        rho = next(r[3] for r in rows if r[3] is not None)
        return cls(name, a[:, 0], a[:, 1], a[:, 2], float(rho), a[:, 3], a[:, 4])


# ============================================================
# FIRE BOUNDARY  (Scandpower / IP guideline form)
# ============================================================

@dataclass
class GuidelineFire:
    """Net absorbed flux for a surface at T_s exposed to a fire whose incident
    heat load q_spec was defined at surface temperature T_ref."""
    eps_flame: float        # flame emissivity
    eps_surf: float         # vessel surface emissivity (absorptivity = emissivity)
    h_flame: float          # flame-to-surface convection coefficient [W/m2K]
    T_ref: float            # surface temperature at which q_spec applies [K]
    t_air_mode: str = "balance"
    # "balance"   : T_f solved so that q_net(T_ref) = q_spec
    # "blackbody" : q_spec is incident black-body radiation, sigma*T_f^4 = q_spec
    _cache: dict = field(default_factory=dict, repr=False)

    def q_rad(self, T_s, T_f):
        return self.eps_surf * SIGMA * (self.eps_flame * T_f**4 - T_s**4)

    def q_conv(self, T_s, T_f):
        return self.h_flame * (T_f - T_s)

    def q_net(self, T_s, T_f):
        return self.q_rad(T_s, T_f) + self.q_conv(T_s, T_f)

    def dq_dTs(self, T_s):
        return -self.h_flame - 4.0 * self.eps_surf * SIGMA * T_s**3

    def flame_temperature(self, q_spec):
        """Fire gas temperature for a specified heat load q_spec [W/m2]."""
        if q_spec <= 0.0:
            return None
        key = round(q_spec, 6)
        if key not in self._cache:
            if self.t_air_mode == "blackbody":
                self._cache[key] = (q_spec / SIGMA) ** 0.25
            else:
                f = lambda Tf: self.q_net(self.T_ref, Tf) - q_spec
                self._cache[key] = brentq(f, self.T_ref, 5000.0)
        return self._cache[key]


# ============================================================
# OUTER BOUNDARY CONDITIONS
#   each returns (q_net, dq_net/dT_s, q_rad, q_conv) in W/m2 for surface temp T_s
# ============================================================

class FireBC:
    """Fire with a specified heat-load time series, converted to a fire gas
    temperature (GuidelineFire). Where the load is zero, falls back to `after`
    (normally AmbientBC)."""

    def __init__(self, fire: GuidelineFire, t_series, q_series, after=None):
        self.fire, self.t, self.q, self.after = fire, np.asarray(t_series), np.asarray(q_series), after
        self.T_flame = None

    def __call__(self, T_s, t):
        q_spec = float(np.interp(t, self.t, self.q))
        Tf = self.fire.flame_temperature(q_spec)
        self.T_flame = Tf
        if Tf is None:
            return self.after(T_s, t) if self.after else (0.0, 0.0, 0.0, 0.0)
        qr, qc = self.fire.q_rad(T_s, Tf), self.fire.q_conv(T_s, Tf)
        return qr + qc, self.fire.dq_dTs(T_s), qr, qc


class PrescribedFluxBC:
    """Net absorbed flux given as a time series (e.g. VessFire's 'Energy' column)."""

    def __init__(self, t_series, q_series, rad_series=None):
        self.t, self.q = np.asarray(t_series), np.asarray(q_series)
        self.qr = None if rad_series is None else np.asarray(rad_series)
        self.T_flame = None

    def __call__(self, T_s, t):
        q = float(np.interp(t, self.t, self.q))
        qr = float(np.interp(t, self.t, self.qr)) if self.qr is not None else q
        return q, 0.0, qr, q - qr


class AmbientBC:
    """Natural/forced convection plus radiation exchange with surroundings at T_env."""

    def __init__(self, T_env: float, eps: float, h: float):
        self.T_env, self.eps, self.h = T_env, eps, h
        self.T_flame = None

    def __call__(self, T_s, t):
        qr = self.eps * SIGMA * (self.T_env**4 - T_s**4)
        qc = self.h * (self.T_env - T_s)
        return qr + qc, -4 * self.eps * SIGMA * T_s**3 - self.h, qr, qc


# ============================================================
# THOMAS ALGORITHM  (from the LNG insulation solver)
# ============================================================

def thomas(lower, diag, upper, rhs):
    n = len(diag)
    d = diag.copy()
    r = rhs.copy()
    for i in range(1, n):
        m = lower[i-1] / d[i-1]
        d[i] -= m * upper[i-1]
        r[i] -= m * r[i-1]
    x = np.empty(n)
    x[-1] = r[-1] / d[-1]
    for i in range(n-2, -1, -1):
        x[i] = (r[i] - upper[i] * x[i+1]) / d[i]
    return x


# ============================================================
# WALL COLUMN
# ============================================================

class WallColumn:
    """Radial conduction through the shell, per metre of vessel length.

    Nodes are vertex-centred: node 0 on the inner surface, node -1 on the outer
    surface, control-volume faces half way between nodes.
    """

    def __init__(self, material: Material, r_inner: float, x_nodes: np.ndarray,
                 outer, T0: float):
        self.mat = material
        self.outer = outer                                      # FireBC / PrescribedFluxBC / AmbientBC
        self.R = r_inner + np.asarray(x_nodes, float)          # node radii
        faces = 0.5 * (self.R[:-1] + self.R[1:])
        self.r_lo = np.concatenate(([self.R[0]], faces))        # CV inner face radius
        self.r_hi = np.concatenate((faces, [self.R[-1]]))       # CV outer face radius
        self.vol = np.pi * (self.r_hi**2 - self.r_lo**2)        # m3 per m length
        self.log_ratio = np.log(self.R[1:] / self.R[:-1])
        self.A_in = 2 * np.pi * self.R[0]                       # m2 per m length
        self.A_out = 2 * np.pi * self.R[-1]
        self.T = np.full(len(self.R), float(T0))
        self.q_net_out = 0.0                                    # W/m2, last step
        self.q_rad_out = 0.0                                    # radiative part
        self.q_conv_out = 0.0                                   # convective part
        self.q_in = 0.0                                         # W/m2 into gas, last step
        self.T_flame = None

    @property
    def T_inner(self):
        return self.T[0]

    @property
    def T_outer(self):
        return self.T[-1]

    def T_mean(self):
        """Volume-weighted through-wall mean temperature."""
        return float(np.sum(self.T * self.vol) / np.sum(self.vol))

    def energy(self):
        """Sensible energy above 0 K per metre length, integrating cp(T)."""
        Tg = np.linspace(0.0, 1600.0, 3201)
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (self.mat.cp_at(Tg[1:]) +
                                                      self.mat.cp_at(Tg[:-1])) * np.diff(Tg))))
        return float(np.sum(self.mat.rho * self.vol * np.interp(self.T, Tg, cum)))

    def step(self, dt: float, t_mid: float, T_gas: float, h_in: float):
        """Advance one implicit time step.

        t_mid  : time at the middle of the step [s] (outer boundary evaluated here)
        T_gas  : gas temperature [K] (lagged)
        h_in   : inner-wall heat transfer coefficient [W/m2K]
        """
        T = self.T
        cp = self.mat.cp_at(T)
        k_face = self.mat.k_at(0.5 * (T[:-1] + T[1:]))
        G = 2 * np.pi * k_face / self.log_ratio                 # W/m/K per m length
        C = self.mat.rho * cp * self.vol / dt                   # W/K per m length

        diag = C.copy()
        diag[:-1] += G
        diag[1:] += G
        lower = -G.copy()
        upper = -G.copy()
        rhs = C * T

        # inner surface: convection to gas
        diag[0] += h_in * self.A_in
        rhs[0] += h_in * self.A_in * T_gas

        # outer surface, linearised about the current surface temperature
        Ts = T[-1]
        q0, dq, _, _ = self.outer(Ts, t_mid)
        diag[-1] += -dq * self.A_out
        rhs[-1] += (q0 - dq * Ts) * self.A_out

        self.T = thomas(lower, diag, upper, rhs)

        self.q_net_out, _, self.q_rad_out, self.q_conv_out = self.outer(self.T[-1], t_mid)
        self.T_flame = self.outer.T_flame
        self.q_in = h_in * (self.T[0] - T_gas)
        return self.q_in


if __name__ == "__main__":
    # quick self-check: a steel slab heated at constant flux conserves energy
    mat = Material.from_vessfire_db("P355NH_P355NL2")
    fire = GuidelineFire(eps_flame=1.0, eps_surf=0.85, h_flame=100.0, T_ref=313.15)
    col = WallColumn(mat, 1.29, VESSFIRE_LOG_NODES_105MM,
                     FireBC(fire, [0, 1e6], [250e3, 250e3]), 313.15)
    E0 = col.energy()
    q_in_tot = 0.0
    for n in range(600):
        col.step(1.0, n + 0.5, 313.15, 0.0)
        q_in_tot += col.q_net_out * col.A_out * 1.0
    dE = col.energy() - E0
    print(f"T_f = {col.T_flame:.1f} K,  T_outer = {col.T_outer-273.15:.1f} C,  "
          f"T_inner = {col.T_inner-273.15:.1f} C after 600 s")
    print(f"energy in {q_in_tot/1e6:.3f} MJ/m, stored {dE/1e6:.3f} MJ/m, "
          f"error {100*(dE-q_in_tot)/q_in_tot:+.2f} %")
