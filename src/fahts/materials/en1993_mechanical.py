"""Mechanical properties of carbon steel at elevated temperature, EN 1993-1-2:2005.

E(T) = 210 GPa * k_E,theta (Table 3.1); thermal elongation (3.4.1.1, eq. 3.1a-c);
nu = 0.3 (EN 1993-1-1 3.2.6, temperature independent).
"""

from __future__ import annotations

import numpy as np


# EN 1993-1-2 Table 3.1, carbon steel: k_E,theta
EC3_T = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200.0])
EC3_KE = np.array([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.31, 0.13, 0.09, 0.0675, 0.045, 0.0225, 0.0])
E20 = 210e3  # MPa, EN 1993-1-1 3.2.6
NU = 0.3  # EN 1993-1-1 3.2.6
E_FLOOR = 1e-3  # keep FE stiffness non-singular at >= 1200 C


def E_of_T(T_C):
    """Young's modulus [MPa], EN 1993-1-2 Table 3.1."""
    return E20 * np.maximum(np.interp(T_C, EC3_T, EC3_KE), E_FLOOR)


def eps_th(T_C):
    """Thermal elongation Delta l/l relative to 20 C, EN 1993-1-2 3.4.1.1."""
    T = np.asarray(T_C, float)
    return np.where(
        T < 750,
        1.2e-5 * T + 0.4e-8 * T**2 - 2.416e-4,
        np.where(T <= 860, 1.1e-2, 2e-5 * T - 6.2e-3),
    )
