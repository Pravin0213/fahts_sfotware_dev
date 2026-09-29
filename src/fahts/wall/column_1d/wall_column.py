"""Radial 1-D transient conduction through a fire-exposed cylindrical wall.

Implicit (backward Euler) vertex-centred finite volume in cylindrical coordinates;
tridiagonal system solved by the Thomas algorithm. Temperature-dependent cp and k.
Outer boundary: any ``fahts.fire`` exposure, linearised about the current surface
temperature. Inner boundary: convection to the contents with h supplied by the caller.

One WallColumn = one radial line of nodes through the shell, representing an area of the
vessel that sees one heat load and one inner condition (e.g. dry / wetted wall).
"""

from __future__ import annotations

import numpy as np

from fahts.materials import SteelTable
from fahts.wall.column_1d.tridiagonal import solve_tridiagonal


class WallColumn:
    """Radial conduction through the shell, per metre of vessel length.

    Nodes are vertex-centred: node 0 on the inner surface, node -1 on the outer
    surface, control-volume faces half way between nodes.
    """

    def __init__(self, material: SteelTable, r_inner: float, x_nodes: np.ndarray,
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

        self.T = solve_tridiagonal(lower, diag, upper, rhs)

        self.q_net_out, _, self.q_rad_out, self.q_conv_out = self.outer(self.T[-1], t_mid)
        self.T_flame = self.outer.T_flame
        self.q_in = h_in * (self.T[0] - T_gas)
        return self.q_in
