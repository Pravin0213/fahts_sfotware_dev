"""
Radiation ball (spherical) prescribed-flux fire source.

Single-zone model: a sphere of radius ``radius`` centred at ``center`` whose
outer surface radiates uniformly at ``flux`` [W/m²] (a Lambertian/diffuse
emitter with constant exitance).  There is no calibration curve to fit — the
incident flux at any point in space follows directly from the exact
point-to-sphere radiative view factor:

    d <= radius   : q = flux                                  (engulfed)
    d >  radius   : q = flux * (radius / d)**2 * cos(theta)    (exterior)

where ``d`` is the distance from the ball centre to the target point and
``theta`` is the angle between the target's outward normal and the direction
from the target toward the ball centre.

Derivation of the two regimes
------------------------------
**Exterior (d > radius):** ``F = (R/d)^2 * cos(theta)`` is the classical
closed-form "differential area to sphere" configuration factor for a point
outside a uniformly-emitting (Lambertian) sphere — see e.g. Incropera,
*Fundamentals of Heat and Mass Transfer*, Table 13.2.  It is *exact*, not a
discretised approximation: a uniformly luminous sphere produces the same
irradiance at any external point as a point source of equal total radiant
power placed at its centre, scaled by the receiver's own cosine law.  Faces
whose outward normal points away from the ball (``cos(theta) <= 0``) receive
zero — this also correctly self-shadows the far side of a convex sphere with
no separate occlusion test needed.

**Interior / engulfed (d <= radius):** a point fully enclosed by a closed
surface subtends the *entire* surface (solid angle 4*pi) regardless of its own
orientation — every ray leaving it in every direction eventually strikes the
enclosing wall.  For a uniform-exitance enclosure this is the standard cavity-
radiation result: irradiance at any interior point equals the wall's own
exitance, isotropically, independent of position or facing direction.  So an
engulfed element receives ``flux`` on *every* exposed face, with no cos(theta)
weighting — this is not a special-cased shortcut, it is the d -> radius limit
of the same physics.

No convective or Stefan-Boltzmann fire term — radiation is fully prescribed by
`flux`.  Set epsilon_m = 0 in the solver when using RadiationBall; steel
re-radiation is applied separately via eps_rerad.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fahts.core.heat.sources.base_source import HeatSource


@dataclass
class RadiationBall(HeatSource):
    """
    Spherical prescribed-flux radiation ball heat source.

    Parameters
    ----------
    name   : identifier string
    center : (3,) array — global coordinates of ball centre [m]
    radius : ball radius [m]
    flux   : uniform prescribed irradiance leaving the ball's outer surface [W/m²]
    active : whether this source participates in the analysis
    """
    name: str
    center: np.ndarray   # shape (3,) [m]
    radius: float        # ball radius [m]
    flux: float          # prescribed surface exitance [W/m²]
    active: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.flux <= 0.0:
            raise ValueError(f"flux must be positive, got {self.flux}")

    # ── HeatSource ABC ────────────────────────────────────────────────────────

    def temperature(self, t: float) -> float:
        """Not applicable for RadiationBall (flux is prescribed); returns ambient 20 °C."""
        return 20.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the sphere [m]."""
        r = self.radius
        return self.center - r, self.center + r

    # ── RadiationBall-specific ────────────────────────────────────────────────

    def max_flux_at_distance(self, distance: float) -> float:
        """
        Direction-agnostic upper bound on incident flux [W/m²] at *distance*
        from the ball centre — the value a directly-facing receiver (or any
        engulfed receiver) would see.  Used for coarse element screening; the
        true per-face value also depends on orientation, see :meth:`incident_flux`.
        """
        if distance <= self.radius:
            return self.flux
        return self.flux * (self.radius / distance) ** 2

    def incident_flux(
        self,
        point: np.ndarray,
        normal: np.ndarray | None = None,
    ) -> float:
        """
        Exact incident flux [W/m²] at *point* with outward unit *normal*.

        ``normal=None`` returns the direction-agnostic magnitude from
        :meth:`max_flux_at_distance` (no cosine clipping applied).
        """
        r_vec = self.center - np.asarray(point, dtype=float)
        d = float(np.linalg.norm(r_vec))
        if d <= self.radius:
            return self.flux
        base = self.flux * (self.radius / d) ** 2
        if normal is None:
            return base
        r_hat = r_vec / d
        cos_theta = float(np.dot(np.asarray(normal, dtype=float), r_hat))
        return base * cos_theta if cos_theta > 0.0 else 0.0

    def exposed_element_ids(
        self,
        elements: dict,
        nodes: dict,
        min_flux: float = 1.0,
    ) -> dict[int, float]:
        """
        Return {eid: max_flux} for beam elements whose midpoint receives more
        than *min_flux* [W/m²] (direction-agnostic upper bound — see
        :meth:`max_flux_at_distance`).  Geometry-only screening; the caller is
        responsible for filtering on ``active`` status.
        """
        result: dict[int, float] = {}
        for beam in elements.values():
            mid = beam.midpoint(nodes)
            dist = float(np.linalg.norm(mid - self.center))
            flux = self.max_flux_at_distance(dist)
            if flux > min_flux:
                result[beam.eid] = flux
        return result
