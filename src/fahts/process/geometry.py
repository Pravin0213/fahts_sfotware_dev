"""Vessel geometry: level <-> volume, wetted and interface areas, characteristic
lengths, stacked free-water + hydrocarbon layers. Horizontal and vertical vessels,
flat / hemispherical / semi-ellipsoidal heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.integrate import quad


# =============================================================================
# 1. GEOMETRY
# =============================================================================
#
# Vessel = cylindrical shell (inner radius R, straight length L) plus two heads
# of depth a:  flat (a = 0), hemispherical (a = R), 2:1 ellipsoidal (a = R/2),
# or any semi-ellipsoid depth a given explicitly.
#
# Horizontal vessel, liquid depth h measured from the bottom of the shell:
#   shell segment area   A_seg(h) = R^2 acos(1 - h/R) - (R - h) sqrt(2Rh - h^2)
#   shell volume         V_shell  = A_seg * L
#   both heads           V_heads  = pi a h^2 (3R - h) / (3R)
#     (exact for semi-ellipsoidal heads with semi-axes R, R, a; hemisphere
#      a = R gives the spherical-cap formula; e.g. D. Jones, "Calculating Tank
#      Volume", 2003; Perry's Handbook 8th ed., Sec. 10 tank-volume formulas)
#   free-surface width   w(h) = 2 sqrt(h (2R - h))
#   interface area       A_i = w L + 2 * (pi/2)(w/2)(a w / (2R))       (= dV/dh)
#   wetted shell area    A_w,shell = 2 R acos(1 - h/R) L
#   wetted head area     A_w,head  = A_head * h / (2R)
#     (exact for a hemisphere: sphere zone area 2 pi R h is linear in h
#      (Archimedes); used as an approximation for ellipsoidal heads)
#
# Vertical vessel, depth h measured from the bottom of the lower head:
#   bottom head (0 <= h <= a):  V = pi R^2 h^2 (3a - h) / (3a^2)
#                               r(h)^2 = R^2 (1 - ((a - h)/a)^2)
#   shell   (a < h < a + L):    V = V_head + pi R^2 (h - a)
#   top head: by symmetry V(h) = V_tot - V_bottom(H_tot - h)
#   wetted head area from the surface-of-revolution integral, parametrised by
#   the polar angle t: z = a(1 - cos t), r = R sin t,
#     dA = 2 pi R sin t sqrt(a^2 sin^2 t + R^2 cos^2 t) dt
#
# The inverse (volume -> level) uses Newton's method with dV/dh = A_i and a
# bisection safeguard.


def _head_area(R: float, a: float) -> float:
    """Inner surface area of one semi-ellipsoidal head (oblate/prolate spheroid
    half-surface), a = head depth. Exact closed forms (e.g. CRC Standard
    Mathematical Tables, spheroid surface area)."""
    if a <= 0:
        return math.pi * R**2
    if abs(a - R) < 1e-12 * R:
        return 2 * math.pi * R**2
    if a < R:  # oblate
        e = math.sqrt(1 - (a / R) ** 2)
        return math.pi * R**2 + math.pi * a**2 / (2 * e) * math.log((1 + e) / (1 - e))
    e = math.sqrt(1 - (R / a) ** 2)  # prolate
    return math.pi * R**2 + math.pi * R * a / e * math.asin(e)


def _cap_area_vertical(R: float, a: float, h: float) -> float:
    """Wetted area of a lower semi-ellipsoidal head filled to depth h (0..a)."""
    if a <= 0:
        return math.pi * R**2 if h > 0 else 0.0
    h = min(max(h, 0.0), a)
    t1 = math.acos(1 - h / a)
    f = (
        lambda t: 2
        * math.pi
        * R
        * math.sin(t)
        * math.sqrt(a**2 * math.sin(t) ** 2 + R**2 * math.cos(t) ** 2)
    )
    return quad(f, 0.0, t1)[0]


@dataclass
class VesselGeometry:
    """Cylindrical vessel with semi-ellipsoidal (incl. hemispherical/flat) heads.

    D        : inner diameter [m]
    L        : straight (tangent-to-tangent) shell length [m]
    orientation : "horizontal" | "vertical"
    head     : "flat" | "hemispherical" | "ellipsoidal" (2:1) | float depth a [m]
    """

    D: float
    L: float
    orientation: str = "horizontal"
    head: object = "ellipsoidal"

    def __post_init__(self):
        self.R = 0.5 * self.D
        if isinstance(self.head, str):
            self.a = {"flat": 0.0, "hemispherical": self.R, "ellipsoidal": self.R / 2}[self.head]
        else:
            self.a = float(self.head)
        self.A_head = _head_area(self.R, self.a)
        self.A_total = 2 * math.pi * self.R * self.L + 2 * self.A_head
        self.V_total = math.pi * self.R**2 * self.L + 4.0 / 3.0 * math.pi * self.R**2 * self.a
        self.H = self.D if self.orientation == "horizontal" else self.L + 2 * self.a

    # ---------------------------------------------------------------- volume
    def volume(self, h: float) -> float:
        """Liquid volume below level h [m3]."""
        R, L, a = self.R, self.L, self.a
        h = min(max(h, 0.0), self.H)
        if self.orientation == "horizontal":
            A_seg = R**2 * math.acos(1 - h / R) - (R - h) * math.sqrt(max(2 * R * h - h * h, 0.0))
            V_heads = math.pi * a * h**2 * (3 * R - h) / (3 * R) if R > 0 else 0.0
            return A_seg * L + V_heads
        # vertical
        V_head = 2.0 / 3.0 * math.pi * R**2 * a

        def v_bottom(y):
            if y <= a:
                return math.pi * R**2 * y**2 * (3 * a - y) / (3 * a**2) if a > 0 else 0.0
            return V_head + math.pi * R**2 * (y - a)

        if h <= a + L:
            return v_bottom(h)
        return self.V_total - v_bottom(self.H - h)

    def interface_area(self, h: float) -> float:
        """Free-surface (gas-liquid) area at level h [m2]; equals dV/dh."""
        R, L, a = self.R, self.L, self.a
        if h <= 0 or h >= self.H:
            return 0.0
        if self.orientation == "horizontal":
            w = 2 * math.sqrt(h * (2 * R - h))
            return w * L + 2 * (math.pi / 2) * (w / 2) * (a * w / (2 * R))
        y = h if h <= a + L else self.H - h
        if y >= a:
            return math.pi * R**2
        return math.pi * R**2 * (1 - ((a - y) / a) ** 2)

    def wetted_area(self, h: float) -> float:
        """Inner wall area below level h [m2]."""
        R, L, a = self.R, self.L, self.a
        h = min(max(h, 0.0), self.H)
        if self.orientation == "horizontal":
            return 2 * R * math.acos(1 - h / R) * L + 2 * self.A_head * h / (2 * R)
        if h >= self.H:
            return self.A_total
        if h <= a:
            return _cap_area_vertical(R, a, h)
        if h <= a + L:
            return self.A_head + 2 * math.pi * R * (h - a)
        return self.A_total - _cap_area_vertical(R, a, self.H - h)

    def wetted_fraction(self, h: float) -> float:
        return self.wetted_area(h) / self.A_total

    def wetted_perimeter_fraction(self, h: float) -> float:
        """Fraction of the shell circumference below the level (horizontal),
        0/1 for vertical shells. = acos(1 - h/R)/pi."""
        if self.orientation != "horizontal":
            return 1.0 if h > self.a else 0.0
        return math.acos(1 - min(max(h, 0.0), self.D) / self.R) / math.pi

    # ---------------------------------------------------------------- inverse
    def level(self, V: float, tol: float = 1e-10) -> float:
        """Level for a liquid volume V (Newton with dV/dh = A_i, bisection guard)."""
        if V <= 0:
            return 0.0
        if V >= self.V_total:
            return self.H
        lo, hi = 0.0, self.H
        h = self.H * V / self.V_total
        for _ in range(100):
            f = self.volume(h) - V
            if abs(f) < tol * self.V_total:
                return h
            if f > 0:
                hi = h
            else:
                lo = h
            dVdh = self.interface_area(h)
            hn = h - f / dVdh if dVdh > 0 else 0.5 * (lo + hi)
            h = hn if lo < hn < hi else 0.5 * (lo + hi)
        return h

    # ---------------------------------------------------------------- lengths
    def interface_char_length(self, h: float) -> float:
        """Goldstein et al. (1973) / Lloyd & Moran (1974) horizontal-surface
        length scale L* = A / P of the free surface."""
        R, L, a = self.R, self.L, self.a
        if h <= 0 or h >= self.H:
            return 0.0
        if self.orientation == "horizontal":
            w = 2 * math.sqrt(h * (2 * R - h))
            Ai = self.interface_area(h)
            # rectangle w x L plus two half-ellipses (Ramanujan perimeter)
            b = a * w / (2 * R)
            ah = w / 2
            per_ell = (
                math.pi * (3 * (ah + b) - math.sqrt((3 * ah + b) * (ah + 3 * b)))
                if (ah + b) > 0
                else 0
            )
            P = 2 * L + per_ell
            return Ai / P
        Ai = self.interface_area(h)
        return math.sqrt(Ai / math.pi) / 2.0  # disc: A/P = r/2

    def liquid_char_length(self, h: float) -> float:
        """Height of the heated wetted wall used for wall->liquid free
        convection (boundary layer runs up the wall to the free surface)."""
        return max(h, 1e-3)

    def gas_char_length(self, h: float) -> float:
        return max(self.H - h, 1e-3)


@dataclass
class LayerState:
    """Geometric description of stacked free-water + hydrocarbon layers."""

    h_water: float
    h_liquid: float  # top of hydrocarbon liquid (>= h_water)
    A_wet_water: float
    A_wet_oil: float
    A_wet_gas: float
    A_int_water_oil: float
    A_int_liquid_gas: float
    V_water: float
    V_oil: float
    V_gas: float


def layers(geom: VesselGeometry, V_water: float, V_oil: float) -> LayerState:
    """Water settles at the bottom, oil above it, gas on top (VessFire's
    published initial state uses a hydrocarbon level plus a free-water level)."""
    hw = geom.level(V_water)
    hl = geom.level(V_water + V_oil)
    Aw = geom.wetted_area(hw)
    Al = geom.wetted_area(hl)
    return LayerState(
        hw,
        hl,
        Aw,
        Al - Aw,
        geom.A_total - Al,
        geom.interface_area(hw) if V_oil > 0 else 0.0,
        geom.interface_area(hl),
        V_water,
        V_oil,
        geom.V_total - V_water - V_oil,
    )


def layers_from_levels(geom: VesselGeometry, h_water: float, h_liquid: float) -> LayerState:
    Vw = geom.volume(h_water)
    return layers(geom, Vw, geom.volume(h_liquid) - Vw)
