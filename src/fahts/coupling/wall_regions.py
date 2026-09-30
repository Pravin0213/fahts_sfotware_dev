"""Split the shell of a horizontal vessel into wall regions: fire zone x contact.

Fire zones (from the heat-load input): a local *peak* zone covering the length fraction
``xi_start..xi_end`` and the arc ``attack_deg +- circ_deg / 2`` (angle measured from the top,
180 = bottom), and the *background* (the rest of the shell). Contact: *wet* = the arc below
the liquid level (centred on the bottom), *dry* = the rest. Heads are not part of the shell
area (flat heads, as in the vessel model).

All fractions are of the shell area pi * D * L and sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeakZone:
    """Local (e.g. jet) fire zone on the shell."""

    xi_start: float  # longitudinal start, fraction of the length [0-1]
    xi_end: float  # longitudinal end [0-1]
    circ_deg: float  # circumferential extent [deg]
    attack_deg: float  # centre angle from the top [deg] (180 = bottom)

    @property
    def length_fraction(self) -> float:
        return min(max(self.xi_end - self.xi_start, 0.0), 1.0)

    @property
    def area_fraction(self) -> float:
        return self.length_fraction * min(max(self.circ_deg, 0.0), 360.0) / 360.0


def _arc_overlap_deg(c1: float, w1: float, c2: float, w2: float) -> float:
    """Overlap [deg] of two arcs on a circle, given centres and full widths [deg]."""
    w1, w2 = min(max(w1, 0.0), 360.0), min(max(w2, 0.0), 360.0)
    if w1 >= 360.0:
        return w2
    if w2 >= 360.0:
        return w1
    best = 0.0
    for shift in (-360.0, 0.0, 360.0):
        lo = max(c1 - w1 / 2, c2 + shift - w2 / 2)
        hi = min(c1 + w1 / 2, c2 + shift + w2 / 2)
        best += max(hi - lo, 0.0)
    return min(best, w1, w2)


def region_fractions(f_wet: float, peak: PeakZone | None) -> dict[str, float]:
    """Area fractions of the regions "dry", "wet" (background) and "peak_dry", "peak_wet".

    f_wet: wetted perimeter fraction (the wetted arc is 360 * f_wet deg, centred on the bottom).
    """
    f_wet = min(max(f_wet, 0.0), 1.0)
    if peak is None or peak.area_fraction <= 0.0:
        return {"dry": 1.0 - f_wet, "wet": f_wet, "peak_dry": 0.0, "peak_wet": 0.0}
    overlap = _arc_overlap_deg(peak.attack_deg, peak.circ_deg, 180.0, 360.0 * f_wet)
    peak_wet = peak.length_fraction * overlap / 360.0
    peak_dry = peak.area_fraction - peak_wet
    wet = max(f_wet - peak_wet, 0.0)
    dry = max(1.0 - peak.area_fraction - wet, 0.0)
    return {"dry": dry, "wet": wet, "peak_dry": peak_dry, "peak_wet": peak_wet}
