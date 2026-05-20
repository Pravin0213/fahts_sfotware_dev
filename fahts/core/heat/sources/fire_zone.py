"""
Phase 2.1 + 2.2 — FireCurve and FireZone.

FireCurve  : time-temperature relationship (ISO 834, hydrocarbon, user-defined).
FireZone   : rectangular box heat source placed in 3-D global space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from fahts.core.heat.sources.base_source import HeatSource


class FireCurveType(Enum):
    ISO_834 = auto()       # Standard cellulosic fire (EN 1991-1-2)
    HYDROCARBON = auto()   # Hydrocarbon / offshore pool fire
    USER_DEFINED = auto()  # Piecewise-linear points supplied by user


@dataclass
class FireCurve:
    """
    Time–temperature relationship for a fire compartment.

    Parameters
    ----------
    curve_type  : one of FireCurveType
    user_points : list of (t [s], T [°C]) pairs — only used for USER_DEFINED
    T_ambient   : ambient temperature [°C] (default 20)

    Formulae (t in minutes):
        ISO 834:     T = T₀ + 345 · log₁₀(8t + 1)
        Hydrocarbon: T = T₀ + 1080 · (1 − 0.325·e^{−0.167t} − 0.675·e^{−2.5t})
        User-def:    piecewise-linear interpolation of supplied (t [s], T [°C]) pairs
    """

    curve_type: FireCurveType = FireCurveType.ISO_834
    user_points: list[tuple[float, float]] = field(default_factory=list)
    T_ambient: float = 20.0

    def temperature(self, t: float) -> float:
        """Return gas temperature [°C] at time *t* [s]."""
        t_min = t / 60.0

        if self.curve_type is FireCurveType.ISO_834:
            if t_min <= 0.0:
                return self.T_ambient
            return self.T_ambient + 345.0 * math.log10(8.0 * t_min + 1.0)

        if self.curve_type is FireCurveType.HYDROCARBON:
            return self.T_ambient + 1080.0 * (
                1.0
                - 0.325 * math.exp(-0.167 * t_min)
                - 0.675 * math.exp(-2.5 * t_min)
            )

        if self.curve_type is FireCurveType.USER_DEFINED:
            if not self.user_points:
                return self.T_ambient
            pts = sorted(self.user_points, key=lambda p: p[0])
            ts = [p[0] for p in pts]
            Ts = [p[1] for p in pts]
            return float(np.interp(t, ts, Ts))

        return self.T_ambient

    def __str__(self) -> str:
        return f"FireCurve({self.curve_type.name}, T₀={self.T_ambient}°C)"


@dataclass
class FireZone(HeatSource):
    """
    Rectangular box heat source in 3-D global coordinate space.

    Attributes
    ----------
    name         : user-visible label
    center       : (x, y, z) geometric centre [m]
    dims         : (dx, dy, dz) full extents [m]
    curve        : FireCurve — T_fire(t) time history
    epsilon_fire : emissivity of fire / hot gas (default 1.0)
    h_conv       : convective heat-transfer coefficient [W/(m²·K)]
                   25 W/(m²·K) for standard (EN 1991-1-2 §3.3.1)
                   50 W/(m²·K) for hydrocarbon
    active       : if False the zone is ignored by the solver

    Exposure rule (Phase 2)
    -----------------------
    A beam element is considered *inside* the fire zone if its midpoint lies
    within the bounding box (optionally plus a tolerance margin).
    All four BOX faces are then fully exposed.
    Partial / edge exposure is deferred to Phase 5 via view-factor geometry.
    """

    name: str
    center: np.ndarray       # shape (3,) [m]
    dims: np.ndarray         # shape (3,) [m]
    curve: FireCurve = field(default_factory=FireCurve)
    epsilon_fire: float = 1.0
    h_conv: float = 25.0
    active: bool = True

    # -- HeatSource interface --------------------------------------------------

    def temperature(self, t: float) -> float:
        """Return fire gas temperature [°C] at time *t* [s]."""
        return self.curve.temperature(t)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) axis-aligned bounding box [m]."""
        c = np.asarray(self.center, dtype=float)
        half = np.asarray(self.dims, dtype=float) / 2.0
        return c - half, c + half

    # -- Geometry helpers ------------------------------------------------------

    def contains_point(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Return True if *point* [m] lies inside the bounding box.

        Parameters
        ----------
        point     : (3,) array in global coordinates [m]
        tolerance : extra margin added symmetrically to each face [m]
        """
        c    = np.asarray(self.center, dtype=float)
        half = np.asarray(self.dims,   dtype=float) / 2.0 + tolerance
        return bool(np.all(np.abs(np.asarray(point, dtype=float) - c) <= half))

    def contains_midpoint(self, midpoint: np.ndarray, tolerance: float = 0.0) -> bool:
        """Convenience alias — test whether a beam midpoint is inside."""
        return self.contains_point(midpoint, tolerance)

    def face_patches(
        self, n_sub: int = 4
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Discretize all 6 box faces into (n_sub × n_sub) sub-patches.

        FAHTS §3.3.4 numerical view factor integration: both source and target
        surfaces are subdivided into small areas for which the simplified
        point-to-point formula F = A_j·cosθ_i·cosθ_j / (π·r²) is valid.

        The *inward* normal of each face points toward the fire zone interior —
        this is the effective emission direction of the hot gas surface.

        Parameters
        ----------
        n_sub : sub-divisions per face edge (default 4 → 16 patches per face)

        Returns
        -------
        list of (centroid [m], inward_normal, area [m²]) tuples
        """
        c = np.asarray(self.center, dtype=float)
        d = np.asarray(self.dims, dtype=float)

        patches: list[tuple[np.ndarray, np.ndarray, float]] = []
        for axis in range(3):
            u_ax = (axis + 1) % 3
            v_ax = (axis + 2) % 3
            u_step = d[u_ax] / n_sub
            v_step = d[v_ax] / n_sub
            sub_area = u_step * v_step

            for sign in (+1, -1):
                face_center = c.copy()
                face_center[axis] += sign * d[axis] / 2.0
                inward = np.zeros(3)
                inward[axis] = -float(sign)  # points toward fire zone interior

                for i in range(n_sub):
                    for j in range(n_sub):
                        patch = face_center.copy()
                        patch[u_ax] += -d[u_ax] / 2.0 + (i + 0.5) * u_step
                        patch[v_ax] += -d[v_ax] / 2.0 + (j + 0.5) * v_step
                        patches.append((patch, inward, sub_area))

        return patches

    def __str__(self) -> str:
        c = self.center
        d = self.dims
        return (
            f"FireZone '{self.name}': "
            f"centre=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m, "
            f"dims=({d[0]:.2f}×{d[1]:.2f}×{d[2]:.2f}) m, "
            f"{self.curve}"
        )
