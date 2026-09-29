"""Tabulated temperature-dependent steel properties for wall conduction and rupture.

The product does not ship property tables yet: the process-model port currently takes
them from the reference data used for calibration (``validation/vessfire/material_db.py``,
local only). Own tables (EN 1993-1-2 / EN 10028 / material certificates) come next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SteelTable:
    """Temperature-dependent steel properties (T in K), linear interpolation."""

    name: str
    T: np.ndarray
    cp: np.ndarray  # J/kg/K
    k: np.ndarray  # W/m/K
    rho: float  # kg/m3 (constant)
    f_yield: np.ndarray  # yield retention factor
    f_uts: np.ndarray  # UTS retention factor

    def cp_at(self, T):
        return np.interp(T, self.T, self.cp)

    def k_at(self, T):
        return np.interp(T, self.T, self.k)

    def f_yield_at(self, T):
        return np.interp(T, self.T, self.f_yield)

    def f_uts_at(self, T):
        return np.interp(T, self.T, self.f_uts)
