"""
Inner (cavity / wetted-surface) Robin boundary condition for 3-D solid members.

Applied on the FACE_INNER faces of a `SolidMesh` by `SolidTransientSolver`:

    −k ∂T/∂n = h(t) · (T − T_fluid(t))        on Γ_inner

This is the hook for future coupling to vessel / pipe contents (hydrocarbon
liquid or gas).  When no InnerRobinBC is given the inner surface is adiabatic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class InnerRobinBC:
    """
    Robin (convective) exchange between the inner wall surface and a fluid.

    Attributes:
        h:       Heat-transfer coefficient [W/(m²·K)]: constant or ``h(t_s)``.
        T_fluid: Fluid (contents) temperature [°C]: constant or ``T(t_s)``.
    """

    h: float | Callable[[float], float]
    T_fluid: float | Callable[[float], float]

    def __post_init__(self) -> None:
        if not callable(self.h) and float(self.h) < 0.0:
            raise ValueError(f"InnerRobinBC.h must be >= 0, got {self.h}")

    def h_at(self, t: float) -> float:
        """Heat-transfer coefficient at time *t* [W/(m²·K)]."""
        return float(self.h(t)) if callable(self.h) else float(self.h)

    def T_fluid_at(self, t: float) -> float:
        """Fluid temperature at time *t* [°C]."""
        return float(self.T_fluid(t)) if callable(self.T_fluid) else float(self.T_fluid)
