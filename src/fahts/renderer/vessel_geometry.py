"""3-D geometry of a horizontal process vessel for display (no Qt, no rendering).

Coordinates: axis along x (0..L), z up, y horizontal; the shell is the steel mid-surface at the
inner radius. Angles are measured from the top (0 = top, 180 = bottom), the convention of the
heat load's peak zone.

Shell cells carry a ``region`` id matching the model's wall regions at a given liquid level:
    0 = "dry" (background, above the liquid)    1 = "wet" (background, below the liquid)
    2 = "peak_dry"                               3 = "peak_wet"
so results can be painted per cell from the model's region temperatures.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from fahts.coupling.vessel_case import VesselCase

REGION_KEYS = ("dry", "wet", "peak_dry", "peak_wet")
REGION_COLUMNS = {"dry": "background", "wet": "wet", "peak_dry": "peak", "peak_wet": "peak_wet"}


def _angle_distance(a, b):
    """Smallest angle between directions a and b [deg]."""
    d = np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)
    return d


def wetted_half_angle_deg(level: float, D: float) -> float:
    """Half-angle of the wetted arc, measured from the bottom [deg]."""
    R = D / 2.0
    if level <= 0.0:
        return 0.0
    if level >= D:
        return 180.0
    return float(np.degrees(np.arccos(np.clip((R - level) / R, -1.0, 1.0))))


class VesselGeometry3D:
    """Shell (with region ids), flat heads and liquid body of one vessel case."""

    def __init__(self, case: VesselCase, n_theta: int = 144, n_length: int = 80,
                 peak_modelled: bool | None = None):
        v, h = case.vessel, case.heat_load
        self.D, self.L = v.inner_diameter_m, v.length_m
        self.R = self.D / 2.0
        p = h.peak
        if peak_modelled is None:
            peak_modelled = (case.options.get("peak_zone", True) and h.has_fire
                             and h.q_background_kW_m2 != h.q_peak_kW_m2
                             and p.xi_end > p.xi_start and p.circ_deg > 0)
        self.peak = p if peak_modelled else None

        # grid lines: uniform, plus the peak-zone boundaries so the zone edges are exact
        th = np.linspace(0.0, 360.0, n_theta + 1)
        xs = np.linspace(0.0, 1.0, n_length + 1)
        if self.peak is not None:
            th = np.unique(np.concatenate(
                (th, np.mod([p.attack_deg - p.circ_deg / 2, p.attack_deg + p.circ_deg / 2], 360))))
            xs = np.unique(np.concatenate((xs, [p.xi_start, p.xi_end])))
        self._th, self._xs = th, xs
        T, X = np.meshgrid(np.radians(th), xs * self.L, indexing="ij")
        grid = pv.StructuredGrid(X, self.R * np.sin(T), self.R * np.cos(T))
        try:
            surface = grid.extract_surface(algorithm="dataset_surface")
        except TypeError:  # pyvista < 0.47 has no algorithm keyword
            surface = grid.extract_surface()
        # merge the duplicated 0/360 deg seam points: a closed shell has no seam edge (the
        # jet-zone outline would otherwise show a line where the zone crosses the top)
        self.shell = surface.clean()
        centres = self.shell.cell_centers().points
        self._cell_theta = np.degrees(np.arctan2(centres[:, 1], centres[:, 2])) % 360.0
        self._cell_xi = centres[:, 0] / self.L
        if self.peak is not None:
            in_len = (self._cell_xi >= p.xi_start) & (self._cell_xi <= p.xi_end)
            in_arc = _angle_distance(self._cell_theta, p.attack_deg) <= p.circ_deg / 2 + 1e-9
            self._in_peak = in_len & in_arc
        else:
            self._in_peak = np.zeros(self.shell.n_cells, bool)
        # cell areas for checks: region areas should match the model's fractions
        self.cell_area = self.shell.compute_cell_sizes(length=False, volume=False)["Area"]
        self.heads = [pv.Disc(center=(x, 0.0, 0.0), inner=0.0, outer=self.R, normal=(1, 0, 0),
                              c_res=n_theta) for x in (0.0, self.L)]
        self.set_level(0.0)

    # ------------------------------------------------------------------ liquid level
    def set_level(self, level: float) -> None:
        """Update region ids of the shell cells for a liquid level [m]."""
        self.level = min(max(level, 0.0), self.D)
        half = wetted_half_angle_deg(self.level, self.D)
        wet = _angle_distance(self._cell_theta, 180.0) < half
        region = np.where(self._in_peak, np.where(wet, 3, 2), np.where(wet, 1, 0))
        self.shell.cell_data["region"] = region.astype(np.int32)

    def region_fractions(self) -> dict[str, float]:
        """Area fraction of each region on the shell at the current level."""
        r = self.shell.cell_data["region"]
        total = self.cell_area.sum()
        return {k: float(self.cell_area[r == i].sum() / total) for i, k in enumerate(REGION_KEYS)}

    def liquid_body(self) -> pv.PolyData | None:
        """Closed liquid volume below the level (None if empty)."""
        if self.level <= 1e-6:
            return None
        half = np.radians(wetted_half_angle_deg(self.level, self.D))
        a = np.linspace(np.pi - half, np.pi + half, 64)      # from the top, around the bottom
        yz = np.column_stack([self.R * np.sin(a), self.R * np.cos(a)])
        pts = np.column_stack([np.zeros(len(yz)), yz])
        face = pv.PolyData(pts, faces=np.r_[len(pts), np.arange(len(pts))]).triangulate()
        return face.extrude((self.L, 0.0, 0.0), capping=True)

    # ------------------------------------------------------------------ results
    def paint(self, temperatures: dict[str, float]) -> None:
        """Cell scalar ``T_C`` from per-region temperatures (keys as in REGION_KEYS)."""
        r = self.shell.cell_data["region"]
        vals = np.array([temperatures.get(k, np.nan) for k in REGION_KEYS])
        self.shell.cell_data["T_C"] = vals[r]


def region_temperatures(series, row: int, which: str = "mean") -> dict[str, float]:
    """Region temperatures [C] at an output row of a model time series.

    which: "mean" (through-wall), "out" (outer surface) or "in" (inner surface).
    """
    out = {}
    for key, col in REGION_COLUMNS.items():
        name = f"{col}_T_{which}_C"
        if name in series:
            out[key] = float(series[name].iloc[row])
    return out
