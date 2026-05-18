"""
Steel material with temperature-dependent thermal properties.
Source: EN 1993-1-2:2005 Annex C (Clause C.4 — thermal properties).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SteelMaterial:
    """
    Isotropic steel material (USFOS MISOIEP card).

    Mechanical properties: E, nu, fy, rho, alpha_T (from .fem file)
    Thermal properties:    k(T) and cp(T) from EN 1993-1-2 Annex C
    """
    mid: int
    E: float        # Young's modulus [Pa]
    nu: float       # Poisson's ratio
    fy: float       # yield stress [Pa]
    rho: float      # density [kg/m³]
    alpha_T: float  # thermal expansion coefficient [1/K]
    name: str = "Steel"

    # ── EN 1993-1-2 thermal properties ───────────────────────────────────────

    def conductivity(self, T_C: float) -> float:
        """
        Thermal conductivity k [W/(m·K)] as function of temperature [°C].
        EN 1993-1-2 §3.4.1.3:
            20 ≤ T ≤ 800:  k = 54 − 3.33×10⁻² T
            800 < T ≤ 1200: k = 27.3
        """
        T = np.clip(T_C, 20.0, 1200.0)
        if T <= 800.0:
            return 54.0 - 3.33e-2 * T
        return 27.3

    def specific_heat(self, T_C: float) -> float:
        """
        Specific heat cp [J/(kg·K)] as function of temperature [°C].
        EN 1993-1-2 §3.4.1.2 (simplified model without phase transformation peak
        for Phase 3; full model with 735°C spike for Phase 5).

        Simplified (used in Phase 3):
            20 ≤ T ≤ 600:  cp = 425 + 7.73×10⁻¹ T − 1.69×10⁻³ T² + 2.22×10⁻⁶ T³
            600 < T ≤ 735: cp = 666 + 13002/(738−T)
            735 < T ≤ 900: cp = 545 + 17820/(T−731)
            900 < T ≤ 1200: cp = 650
        """
        T = float(np.clip(T_C, 20.0, 1200.0))
        if T <= 600.0:
            return (425.0 + 7.73e-1 * T - 1.69e-3 * T**2 + 2.22e-6 * T**3)
        if T <= 735.0:
            denom = 738.0 - T
            return 666.0 + 13002.0 / denom if denom > 0.1 else 1e6
        if T <= 900.0:
            denom = T - 731.0
            return 545.0 + 17820.0 / denom if denom > 0.1 else 1e6
        return 650.0

    def density(self, T_C: float = 20.0) -> float:  # noqa: ARG002
        """Density [kg/m³] — treated as constant for steel (EN 1993-1-2 §3.4.1.1)."""
        return self.rho

    def __str__(self) -> str:
        return (f"SteelMaterial {self.mid} ({self.name}): "
                f"E={self.E/1e9:.0f} GPa, fy={self.fy/1e6:.0f} MPa, "
                f"ρ={self.rho:.0f} kg/m³")
