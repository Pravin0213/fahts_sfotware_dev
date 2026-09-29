"""
Insulation layer model — FAHTS §3.3.3.

Models a massless (type 1) thermal-resistance insulation layer between the fire
environment and the steel surface.  The layer has a temperature-dependent
conductivity λ_i(T) and thickness d_i, producing a series thermal resistance:

    R_ins = d_i / λ_i(T_mean)   [m²·K/W]

The equivalent heat flux through the insulation is:

    q_ins = λ_i(T_mean) / d_i × (T_fire − T_steel)

where T_mean = (T_fire + T_steel) / 2 is the mean temperature across the
insulation thickness.  This replaces the direct convective/radiative boundary
with an insulation-driven boundary at each Picard iterate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class InsulationLayer:
    """
    Massless (type 1) thermal-resistance insulation layer.

    The layer is modelled as a pure thermal resistance; its own heat storage is
    neglected (density = 0, specific_heat = 0).

    Attributes:
        thickness:      Insulation thickness d_i [m].  Must be > 0.
        conductivity:   Thermal conductivity λ_i [W/(m·K)].  Either a constant
                        float or a callable ``λ(T_mean_degC) → float``.
        density:        Layer material density [kg/m³].  Zero = massless (default).
        specific_heat:  Layer material specific heat [J/(kg·K)].  Zero = massless.
    """

    thickness: float
    conductivity: float | Callable[[float], float]
    density: float = 0.0
    specific_heat: float = 0.0

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError(
                f"InsulationLayer.thickness must be positive, got {self.thickness}"
            )
        if isinstance(self.conductivity, (int, float)) and self.conductivity <= 0.0:
            raise ValueError(
                f"InsulationLayer.conductivity must be positive, got {self.conductivity}"
            )
        if self.density < 0.0:
            raise ValueError(
                f"InsulationLayer.density must be >= 0, got {self.density}"
            )
        if self.specific_heat < 0.0:
            raise ValueError(
                f"InsulationLayer.specific_heat must be >= 0, got {self.specific_heat}"
            )

    # ── Physics helpers ───────────────────────────────────────────────────────

    def effective_conductivity(self, T_mean: float) -> float:
        """
        Return λ_i evaluated at *T_mean* [°C].

        If conductivity was supplied as a constant float, that value is returned
        regardless of temperature.  If supplied as a callable, it is evaluated.

        Args:
            T_mean: Mean temperature across the insulation layer [°C].

        Returns:
            Thermal conductivity [W/(m·K)].
        """
        if callable(self.conductivity):
            return float(self.conductivity(T_mean))
        return float(self.conductivity)

    def resistance(self, T_mean: float) -> float:
        """
        Return the thermal resistance d_i / λ_i(T_mean) [m²·K/W].

        Args:
            T_mean: Mean temperature across the insulation layer [°C].

        Returns:
            Thermal resistance [m²·K/W].
        """
        lam = self.effective_conductivity(T_mean)
        if lam <= 0.0:
            raise ValueError(
                f"Insulation conductivity must be positive at T_mean={T_mean:.1f} °C, "
                f"got {lam}"
            )
        return self.thickness / lam

    def conductance(self, T_mean: float) -> float:
        """
        Return the insulation surface conductance h_ins = λ_i / d_i [W/(m²·K)].

        This is added as a linearised convective-like stiffness term into the
        solver's K_e matrix in place of the standard h_conv term.

        Args:
            T_mean: Mean temperature across the insulation layer [°C].

        Returns:
            Insulation conductance [W/(m²·K)].
        """
        return 1.0 / self.resistance(T_mean)
