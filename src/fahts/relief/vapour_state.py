"""Per-mass real-gas (PR) vapour properties at (T, P), with derivatives for nozzle and
line integration.
"""

from __future__ import annotations


class VapourState:
    """Per-mass vapour properties at (T, P) from the PR vapour root."""

    __slots__ = ("T", "P", "v", "h", "s", "cp", "cv", "vT", "vP", "w", "Z", "M", "rho_mol")

    def __init__(self, m, x, T, P):
        p = m.phase_props(x, T, P, "V")
        M = p.M
        self.T, self.P, self.M, self.Z = T, P, M, p.Z
        self.v = p.v / M
        self.h = p.h / M
        self.s = p.s / M
        self.cp = p.cp / M
        self.cv = p.cv / M
        self.vT = -p.dPdT_v / p.dPdv_T / M  # (dv/dT)_P  per kg
        self.vP = 1.0 / p.dPdv_T / M  # (dv/dP)_T  per kg
        self.w = p.w
        self.rho_mol = 1.0 / (p.v + x @ m._c)
