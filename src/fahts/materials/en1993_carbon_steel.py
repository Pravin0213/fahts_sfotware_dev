"""Built-in carbon steel table from EN 1993-1-2:2005 (Eurocode 3, structural fire design).

- specific heat cp(T): clause 3.4.1.2, eq. (3.2a-d), including the ~735 C transformation peak
- thermal conductivity k(T): clause 3.4.1.3, eq. (3.3a-b)
- density 7850 kg/m3: clause 3.2.2
- yield strength retention k_y,theta: Table 3.1 (effective yield strength)
- ultimate strength at temperature, Annex A (strain hardening allowed):
  f_u,theta = 1.25 f_y,theta (T < 300 C), f_y,theta (2 - 0.0025 T) (300-400 C), f_y,theta (T >= 400 C);
  retention relative to room temperature f_u,20 = 1.25 f_y.

A generic default, not a specific grade: for design work supply the grade's own data
(``SteelTable.from_csv``).
"""

from __future__ import annotations

import numpy as np

from fahts.materials.steel_table import SteelTable

# EN 1993-1-2 Table 3.1: effective yield strength reduction factor k_y,theta
_T_KY = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200.0])
_KY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.78, 0.47, 0.23, 0.11, 0.06, 0.04, 0.02, 0.0])


def cp_en1993(T_C):
    """Specific heat [J/kg/K], EN 1993-1-2 eq. (3.2)."""
    T = np.asarray(T_C, float)
    with np.errstate(divide="ignore"):
        return np.select(
            [T < 600.0, T < 735.0, T < 900.0],
            [
                425.0 + 0.773 * T - 1.69e-3 * T**2 + 2.22e-6 * T**3,
                666.0 + 13002.0 / (738.0 - T),
                545.0 + 17820.0 / (T - 731.0),
            ],
            650.0,
        )


def k_en1993(T_C):
    """Thermal conductivity [W/m/K], EN 1993-1-2 eq. (3.3)."""
    T = np.asarray(T_C, float)
    return np.where(T < 800.0, 54.0 - 3.33e-2 * T, 27.3)


def ky_en1993(T_C):
    """Effective yield strength retention k_y,theta, EN 1993-1-2 Table 3.1."""
    return np.interp(T_C, _T_KY, _KY)


def ku_en1993(T_C):
    """Ultimate strength retention f_u,theta / f_u,20 (Annex A, f_u,20 = 1.25 f_y)."""
    T = np.asarray(T_C, float)
    hardening = np.where(T < 300.0, 1.25, np.where(T < 400.0, 2.0 - 0.0025 * T, 1.0))
    return ky_en1993(T) * hardening / 1.25


def en1993_carbon_steel(name: str = "EN 1993-1-2 carbon steel") -> SteelTable:
    """Carbon steel property table from 20 to 1200 C (resolves the 735 C cp peak)."""
    T_C = np.unique(np.concatenate((np.arange(20.0, 1201.0, 5.0), np.arange(720.0, 750.0, 0.5))))
    return SteelTable(
        name=name,
        T=T_C + 273.15,
        cp=cp_en1993(T_C),
        k=k_en1993(T_C),
        rho=7850.0,
        f_yield=ky_en1993(T_C),
        f_uts=ku_en1993(T_C),
    )
