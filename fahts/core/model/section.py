from __future__ import annotations
import math
from dataclasses import dataclass
from abc import ABC


class Section(ABC):
    """Abstract base for cross-section types."""
    sid: int


@dataclass
class BoxSection(Section):
    """
    Hollow rectangular (BOX) cross-section — USFOS BOX card.

    Dimensions in metres:
        H      — overall height (local y-direction)
        W      — overall width  (local z-direction)
        T_side — side wall thickness (left and right)
        T_bot  — bottom plate thickness
        T_top  — top plate thickness
    """
    sid: int
    H: float       # height [m]
    T_side: float  # side wall thickness [m]
    T_bot: float   # bottom plate thickness [m]
    T_top: float   # top plate thickness [m]
    W: float       # width [m]

    @property
    def outer_perimeter(self) -> float:
        """Outer perimeter of the BOX [m]."""
        return 2.0 * (self.H + self.W)

    @property
    def inner_height(self) -> float:
        return max(0.0, self.H - self.T_bot - self.T_top)

    @property
    def inner_width(self) -> float:
        return max(0.0, self.W - 2.0 * self.T_side)

    @property
    def cross_section_area(self) -> float:
        """Steel area of the cross-section [m²]."""
        A_outer = self.H * self.W
        A_inner = self.inner_height * self.inner_width
        return A_outer - A_inner

    @property
    def section_factor_Am_V(self) -> float:
        """
        Section factor A_m / V  [1/m] — exposed steel surface per unit volume.
        Used in EN 1993-1-2 lumped thermal mass approach (Phase 2 interim).
        Assumes all four outer faces are exposed.
        """
        A_m = self.outer_perimeter   # exposed perimeter per unit length [m]
        V   = self.cross_section_area  # steel area per unit length [m²]
        return A_m / V if V > 0 else 0.0

    def __str__(self) -> str:
        return (f"BOX {self.sid}: "
                f"H={self.H*1000:.0f}×W={self.W*1000:.0f} mm, "
                f"T_side={self.T_side*1000:.0f} mm, "
                f"T_bot={self.T_bot*1000:.0f} mm, "
                f"T_top={self.T_top*1000:.0f} mm")


@dataclass
class PipeSection(Section):
    """
    Circular hollow (PIPE) cross-section — USFOS PIPE card.

    Dimensions in metres:
        outer_diameter — outside diameter
        thickness      — wall thickness
    """
    sid: int
    outer_diameter: float  # [m]
    thickness: float       # wall thickness [m]

    @property
    def inner_diameter(self) -> float:
        return max(0.0, self.outer_diameter - 2.0 * self.thickness)

    @property
    def outer_radius(self) -> float:
        return self.outer_diameter / 2.0

    @property
    def inner_radius(self) -> float:
        return self.inner_diameter / 2.0

    @property
    def cross_section_area(self) -> float:
        """Steel area of the annular cross-section [m²]."""
        return math.pi / 4.0 * (self.outer_diameter ** 2 - self.inner_diameter ** 2)

    @property
    def outer_perimeter(self) -> float:
        return math.pi * self.outer_diameter

    @property
    def section_factor_Am_V(self) -> float:
        V = self.cross_section_area
        return self.outer_perimeter / V if V > 0 else 0.0

    def __str__(self) -> str:
        return (f"PIPE {self.sid}: "
                f"OD={self.outer_diameter*1000:.0f} mm, "
                f"t={self.thickness*1000:.0f} mm")


@dataclass
class ISection(Section):
    """
    I-beam cross-section — USFOS IHPROFIL card.

    Dimensions in metres:
        h      — total section height
        tw     — web thickness
        bf_top — top flange width
        tf_top — top flange thickness
        bf_bot — bottom flange width
        tf_bot — bottom flange thickness
    """
    sid: int
    h: float       # total height [m]
    tw: float      # web thickness [m]
    bf_top: float  # top flange width [m]
    tf_top: float  # top flange thickness [m]
    bf_bot: float  # bottom flange width [m]
    tf_bot: float  # bottom flange thickness [m]

    @property
    def web_height(self) -> float:
        """Clear web height (between flanges) [m]."""
        return max(0.0, self.h - self.tf_top - self.tf_bot)

    @property
    def cross_section_area(self) -> float:
        """Steel area [m²]."""
        return (self.bf_top * self.tf_top
                + self.tw * self.web_height
                + self.bf_bot * self.tf_bot)

    @property
    def outer_perimeter(self) -> float:
        """Approximate exposed perimeter [m] for section-factor calculation."""
        # Both flanges top+bottom, web sides exposed (two sides)
        return (2.0 * self.bf_top + 2.0 * self.bf_bot
                + 2.0 * self.web_height
                + 2.0 * (self.tf_top + self.tf_bot))

    @property
    def section_factor_Am_V(self) -> float:
        V = self.cross_section_area
        return self.outer_perimeter / V if V > 0 else 0.0

    def __str__(self) -> str:
        return (f"IHPROFIL {self.sid}: "
                f"h={self.h*1000:.0f} mm, tw={self.tw*1000:.0f} mm, "
                f"bf={self.bf_top*1000:.0f}/{self.bf_bot*1000:.0f} mm")


@dataclass
class PlateSection(Section):
    """
    Flat plate/shell thickness — USFOS PLTHICK card.
    Used by QUADSHEL and TRISHELL elements.
    """
    sid: int
    thickness: float  # [m]

    def __str__(self) -> str:
        return f"PLTHICK {self.sid}: t={self.thickness*1000:.1f} mm"
