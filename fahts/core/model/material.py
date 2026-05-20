"""
Steel material with temperature-dependent thermal properties.

Two property modes are supported:

1. EN 1993-1-2 Annex C (default, usfos_mode=False)
   Piecewise polynomial formulas from EN 1993-1-2:2005 §3.4.1.2/3.
   Used in FAHTS production runs.

2. USFOS reference mode (usfos_mode=True)
   Constant base values multiplied by temperature-dependent factors from
   the USFOS thermpar × tempdepy tables in usfos_verification_results/fahts.fem:
       thermpar: rho=7850, c_ref=510 J/kg·K, k_ref=50 W/m·K, emiss=0.85
       tempdepy 100 (cp factors):  T=[0..1300], f=[0.792..1.282]
       tempdepy 200 (k  factors):  T=[0..1300], f=[1.084..0.548]
   Used only for benchmark comparison against the USFOS reference output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── USFOS thermpar × tempdepy tables (from fahts.fem) ────────────────────────
# tempdepy 100 — specific heat multiplier (c_ref = 510 J/kg·K)
_USFOS_CP_T: np.ndarray = np.array(
    [0, 100, 200, 300, 400, 500, 600, 650, 685, 731, 750, 773, 807, 870, 1300],
    dtype=float,
)
_USFOS_CP_F: np.ndarray = np.array(
    [0.792, 0.943, 1.018, 1.094, 1.131, 1.282, 1.508, 1.584,
     1.697, 9.804, 2.790, 1.998, 1.471, 1.282, 1.282],
    dtype=float,
)

# tempdepy 200 — conductivity multiplier (k_ref = 50 W/m·K)
_USFOS_K_T: np.ndarray = np.array(
    [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300],
    dtype=float,
)
_USFOS_K_F: np.ndarray = np.array(
    [1.084, 1.019, 0.949, 0.874, 0.809, 0.744, 0.679,
     0.614, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548],
    dtype=float,
)

_USFOS_C_REF: float = 510.0   # J/(kg·K)
_USFOS_K_REF: float = 50.0    # W/(m·K)


@dataclass
class SteelMaterial:
    """
    Isotropic steel material (USFOS MISOIEP card).

    Mechanical properties: E, nu, fy, rho, alpha_T (from .fem file).
    Thermal properties:    k(T) and cp(T) from EN 1993-1-2 Annex C (default)
                           or USFOS thermpar × tempdepy tables (usfos_mode=True).

    Parameters
    ----------
    usfos_mode : bool
        When True, conductivity() and specific_heat() use the piecewise-linear
        tables from usfos_verification_results/fahts.fem instead of the Annex C
        polynomial formulas.  Set this flag only for benchmark comparisons.
    """
    mid: int
    E: float        # Young's modulus [Pa]
    nu: float       # Poisson's ratio
    fy: float       # yield stress [Pa]
    rho: float      # density [kg/m³]
    alpha_T: float  # thermal expansion coefficient [1/K]
    name: str = "Steel"
    usfos_mode: bool = field(default=False, compare=False, repr=False)

    # ── EN 1993-1-2 Annex C thermal properties ────────────────────────────────

    def _conductivity_ec3(self, T_C: float) -> float:
        """
        Thermal conductivity k [W/(m·K)] from EN 1993-1-2 §3.4.1.3.
            20 ≤ T ≤ 800:   k = 54 − 3.33×10⁻² T
            800 < T ≤ 1200: k = 27.3
        """
        T = float(np.clip(T_C, 20.0, 1200.0))
        if T <= 800.0:
            return 54.0 - 3.33e-2 * T
        return 27.3

    def _specific_heat_ec3(self, T_C: float) -> float:
        """
        Specific heat cp [J/(kg·K)] from EN 1993-1-2 §3.4.1.2.
            20 ≤ T ≤ 600:  cp = 425 + 7.73×10⁻¹ T − 1.69×10⁻³ T² + 2.22×10⁻⁶ T³
            600 < T ≤ 735: cp = 666 + 13002/(738−T)
            735 < T ≤ 900: cp = 545 + 17820/(T−731)
            900 < T ≤ 1200: cp = 650
        """
        T = float(np.clip(T_C, 20.0, 1200.0))
        if T <= 600.0:
            return 425.0 + 7.73e-1 * T - 1.69e-3 * T ** 2 + 2.22e-6 * T ** 3
        if T <= 735.0:
            denom = 738.0 - T
            return 666.0 + 13002.0 / denom if denom > 0.1 else 1e6
        if T <= 900.0:
            denom = T - 731.0
            return 545.0 + 17820.0 / denom if denom > 0.1 else 1e6
        return 650.0

    # ── USFOS thermpar × tempdepy tables ──────────────────────────────────────

    def _conductivity_usfos(self, T_C: float) -> float:
        """
        Thermal conductivity k [W/(m·K)] from USFOS fahts.fem thermpar tables.
        k(T) = k_ref × interp(tempdepy_200, T)   where k_ref = 50 W/m·K.
        Valid range: 0–1300 °C; clipped outside.
        """
        T = float(np.clip(T_C, 0.0, 1300.0))
        factor = float(np.interp(T, _USFOS_K_T, _USFOS_K_F))
        return _USFOS_K_REF * factor

    def _specific_heat_usfos(self, T_C: float) -> float:
        """
        Specific heat cp [J/(kg·K)] from USFOS fahts.fem thermpar tables.
        cp(T) = c_ref × interp(tempdepy_100, T)  where c_ref = 510 J/kg·K.
        Valid range: 0–1300 °C; clipped outside.
        """
        T = float(np.clip(T_C, 0.0, 1300.0))
        factor = float(np.interp(T, _USFOS_CP_T, _USFOS_CP_F))
        return _USFOS_C_REF * factor

    # ── Public API ────────────────────────────────────────────────────────────

    def conductivity(self, T_C: float) -> float:
        """
        Thermal conductivity k [W/(m·K)] at temperature T_C [°C].

        Dispatches to the USFOS thermpar table (usfos_mode=True) or the
        EN 1993-1-2 Annex C formula (default).
        """
        if self.usfos_mode:
            return self._conductivity_usfos(T_C)
        return self._conductivity_ec3(T_C)

    def specific_heat(self, T_C: float) -> float:
        """
        Specific heat cp [J/(kg·K)] at temperature T_C [°C].

        Dispatches to the USFOS thermpar table (usfos_mode=True) or the
        EN 1993-1-2 Annex C formula (default).
        """
        if self.usfos_mode:
            return self._specific_heat_usfos(T_C)
        return self._specific_heat_ec3(T_C)

    def density(self, T_C: float = 20.0) -> float:  # noqa: ARG002
        """Density [kg/m³] — constant (EN 1993-1-2 §3.4.1.1)."""
        return self.rho

    def __str__(self) -> str:
        mode = "USFOS" if self.usfos_mode else "EC3"
        return (
            f"SteelMaterial {self.mid} ({self.name}, {mode}): "
            f"E={self.E / 1e9:.0f} GPa, fy={self.fy / 1e6:.0f} MPa, "
            f"ρ={self.rho:.0f} kg/m³"
        )
