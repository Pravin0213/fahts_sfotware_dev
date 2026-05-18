"""
Radiation ball (spherical) fire source — USFOS USERFLUX type 0.

Two-zone prescribed flux source:
  - Inner zone (distance ≤ r1): flux = flux1 [W/m²]
  - Outer zone (r1 < distance ≤ r2): flux = flux2 [W/m²]
  - Beyond r2: not exposed (flux = 0)

No convective or Stefan-Boltzmann term — radiation is fully prescribed by the
flux values. Set epsilon_m = 0 in the solver when using RadiationBall.

USFOS USERFLUX format:
    USERFLUX  0  set  cx  cy  cz  r1  flux1  r2  flux2
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fahts.core.heat.sources.base_source import HeatSource


@dataclass
class RadiationBall(HeatSource):
    """
    Spherical two-zone radiation ball heat source (USFOS USERFLUX type 0).

    Parameters
    ----------
    name   : identifier string
    center : (3,) array — global coordinates of ball centre [m]
    r1     : inner zone radius [m]; elements at distance ≤ r1 receive flux1
    flux1  : prescribed irradiance for inner zone [W/m²]
    r2     : outer zone radius [m]; elements at r1 < distance ≤ r2 receive flux2
    flux2  : prescribed irradiance for outer zone [W/m²]
    active : whether this source participates in the analysis
    """
    name: str
    center: np.ndarray   # shape (3,) [m]
    r1: float            # inner zone radius [m]
    flux1: float         # inner zone prescribed flux [W/m²]
    r2: float            # outer zone radius [m]
    flux2: float         # outer zone prescribed flux [W/m²]
    active: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.r1 <= 0:
            raise ValueError(f"r1 must be positive, got {self.r1}")
        if self.r2 <= self.r1:
            raise ValueError(f"r2 ({self.r2}) must be greater than r1 ({self.r1})")

    # ── HeatSource ABC ────────────────────────────────────────────────────────

    def temperature(self, t: float) -> float:
        """Not applicable for RadiationBall (flux is prescribed); returns ambient 20 °C."""
        return 20.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the outer sphere [m]."""
        r = self.r2
        return self.center - r, self.center + r

    # ── RadiationBall-specific ────────────────────────────────────────────────

    def flux_at(self, distance: float) -> float:
        """
        Return prescribed flux [W/m²] for an element at *distance* from centre.

        Returns flux1 for distance ≤ r1, flux2 for r1 < distance ≤ r2, 0 beyond r2.
        """
        if distance <= self.r1:
            return self.flux1
        if distance <= self.r2:
            return self.flux2
        return 0.0

    def exposed_element_ids(
        self,
        elements: dict,
        nodes: dict,
    ) -> dict[int, float]:
        """
        Return {eid: flux} for beam elements whose midpoints fall within r2.

        The flux value is zone-dependent (flux1 or flux2). Elements beyond r2
        are not included in the result.
        """
        result: dict[int, float] = {}
        for beam in elements.values():
            mid = beam.midpoint(nodes)
            dist = float(np.linalg.norm(mid - self.center))
            flux = self.flux_at(dist)
            if flux > 0.0:
                result[beam.eid] = flux
        return result
