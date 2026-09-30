"""3-D geometry of a horizontal process vessel for display (no Qt, no rendering).

Coordinates: axis along x (0..L), z up, y horizontal. ``shell`` is the thin inner surface (the
reference for region areas); ``wall_surface()`` gives the solid steel wall between the inner
radius and inner radius + thickness (optionally exaggerated for display), coloured per cell
from the model's region temperatures, uniform or through the thickness. Angles are measured
from the top (0 = top, 180 = bottom), the convention of the heat load's peak zone.

Shell cells carry a ``region`` id matching the model's wall regions at a given liquid level:
    0 = "dry" (background, above the liquid)    1 = "wet" (background, below the liquid)
    2 = "peak_dry"                               3 = "peak_wet"
so results can be painted per cell from the model's region temperatures.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from fahts.wall.fem_3d.shell_mesh import _merge_close

from fahts.coupling.vessel_case import VesselCase

REGION_KEYS = ("dry", "wet", "peak_dry", "peak_wet")
CUTAWAYS = {"none": None, "quarter": (0.0, 90.0), "half": (0.0, 180.0)}  # removed arc, from top


def _angle_distance(a, b):
    """Smallest angle between directions a and b [deg]."""
    d = np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)
    return d


def _surface(ds) -> pv.PolyData:
    """Outer surface of a dataset (pyvista >= 0.47 wants the algorithm named explicitly)."""
    try:
        return ds.extract_surface(algorithm="dataset_surface")
    except TypeError:  # older pyvista: no algorithm keyword
        return ds.extract_surface()


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
                 peak_modelled: bool | None = None, thickness_scale: float = 1.0,
                 n_radial: int = 8):
        v, h = case.vessel, case.heat_load
        self.D, self.L = v.inner_diameter_m, v.length_m
        self.R = self.D / 2.0
        self.t = v.wall_m
        self.thickness_scale = thickness_scale
        self._n_radial = n_radial
        self.wall = None  # solid wall grid, built on first use (wall_surface)
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
            # merge near-duplicates (a zone edge on a grid line to round-off would leave a
            # sliver), keep the closing 360 deg line for the structured grid
            edges = np.mod([p.attack_deg - p.circ_deg / 2, p.attack_deg + p.circ_deg / 2], 360)
            th = np.append(_merge_close(np.concatenate((th[:-1], edges)), 1e-6, period=360.0),
                           360.0)
            xs = _merge_close(np.concatenate((xs, [p.xi_start, p.xi_end])), 1e-9)
        self._th, self._xs = th, xs
        T, X = np.meshgrid(np.radians(th), xs * self.L, indexing="ij")
        grid = pv.StructuredGrid(X, self.R * np.sin(T), self.R * np.cos(T))
        surface = _surface(grid)
        # merge the duplicated 0/360 deg seam points: a closed shell has no seam edge (the
        # jet-zone outline would otherwise show a line where the zone crosses the top)
        # tolerance: the 0 and 360 deg seam points differ by round-off (sin 2pi = -2e-16)
        self.shell = surface.clean(tolerance=1e-9, absolute=True)
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
        self.R_out = self.R + self.t * thickness_scale  # displayed outer radius
        self.heads = [pv.Disc(center=(x, 0.0, 0.0), inner=0.0, outer=self.R_out,
                              normal=(1, 0, 0), c_res=n_theta) for x in (0.0, self.L)]
        self.set_level(0.0)

    # ------------------------------------------------------------------ liquid level
    def set_level(self, level: float) -> None:
        """Update region ids of the shell cells for a liquid level [m]."""
        self.level = min(max(level, 0.0), self.D)
        half = wetted_half_angle_deg(self.level, self.D)
        wet = _angle_distance(self._cell_theta, 180.0) < half
        region = np.where(self._in_peak, np.where(wet, 3, 2), np.where(wet, 1, 0))
        self.shell.cell_data["region"] = region.astype(np.int32)
        if self.wall is not None:
            self._set_wall_regions(half)

    # ------------------------------------------------------------------ solid wall
    def _build_wall(self) -> None:
        """Solid wall: (theta x length x radial) hexahedra, seam merged, cell geometry cached."""
        th, xs = np.radians(self._th), self._xs * self.L
        r = self.R + np.linspace(0.0, self.t * self.thickness_scale, self._n_radial + 1)
        # axis order (x, theta, r) gives right-handed hexahedra (positive volume, outward
        # surface normals); (theta, x, r) would turn every cell inside out
        X, T, Rr = np.meshgrid(xs, th, r, indexing="ij")
        grid = pv.StructuredGrid(X, Rr * np.sin(T), Rr * np.cos(T))
        # merge the seam (round-off tolerance): no internal seam faces
        self.wall = grid.cast_to_unstructured_grid().clean(tolerance=1e-9)
        c = self.wall.cell_centers().points
        self._w_theta = np.degrees(np.arctan2(c[:, 1], c[:, 2])) % 360.0
        self._w_xi = c[:, 0] / self.L
        if self.peak is not None:
            p = self.peak
            self._w_peak = ((self._w_xi >= p.xi_start) & (self._w_xi <= p.xi_end)
                            & (_angle_distance(self._w_theta, p.attack_deg)
                               <= p.circ_deg / 2 + 1e-9))
        else:
            self._w_peak = np.zeros(self.wall.n_cells, bool)
        self._set_wall_regions(wetted_half_angle_deg(self.level, self.D))

    def _set_wall_regions(self, half: float) -> None:
        wet = _angle_distance(self._w_theta, 180.0) < half
        region = np.where(self._w_peak, np.where(wet, 3, 2), np.where(wet, 1, 0))
        self.wall.cell_data["region"] = region.astype(np.int32)

    def wall_surface(self, cutaway: str = "none") -> pv.PolyData:
        """Visible surface of the solid wall (outer, inner, ends and cut faces) with the
        wall's cell data (``region``)."""
        if self.wall is None:
            self._build_wall()
        cut = CUTAWAYS[cutaway]
        if cut is None:
            solid = self.wall
        else:
            keep = ~((self._w_theta > cut[0]) & (self._w_theta < cut[1]))
            solid = self.wall.extract_cells(np.flatnonzero(keep))
        return _surface(solid)

    def region_fractions(self) -> dict[str, float]:
        """Area fraction of each region on the shell at the current level."""
        r = self.shell.cell_data["region"]
        total = self.cell_area.sum()
        return {k: float(self.cell_area[r == i].sum() / total) for i, k in enumerate(REGION_KEYS)}

    def liquid_body(self, cutaway: str = "none") -> pv.PolyData | None:
        """Closed liquid volume below the level (None if empty); the half cut-away also cuts
        the liquid (the quarter one removes only the upper front, above most liquid)."""
        if self.level <= 1e-6:
            return None
        half = np.radians(wetted_half_angle_deg(self.level, self.D))
        a = np.linspace(np.pi - half, np.pi + half, 64)      # from the top, around the bottom
        yz = np.column_stack([self.R * np.sin(a), self.R * np.cos(a)])
        pts = np.column_stack([np.zeros(len(yz)), yz])
        face = pv.PolyData(pts, faces=np.r_[len(pts), np.arange(len(pts))]).triangulate()
        body = face.extrude((self.L, 0.0, 0.0), capping=True)
        if cutaway == "half":
            body = _surface(body.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True))
        return body

class FieldGeometry3D:
    """The 3-D wall solver's own mesh (``wall.fem_3d.VesselShellMesh``) for display: node
    temperatures as point data, so colours vary smoothly around, along and through the wall.
    ``thickness_scale`` exaggerates the drawn thickness only."""

    def __init__(self, mesh, thickness_scale: float = 1.0):
        self.mesh = mesh
        R = mesh.D / 2.0
        pts = mesh.nodes.copy()
        r = np.hypot(pts[:, 1], pts[:, 2])
        f = (R + (r - R) * thickness_scale) / r
        pts[:, 1] *= f
        pts[:, 2] *= f
        cells = np.hstack([np.full((mesh.n_hexes, 1), 8), mesh.hexes]).ravel()
        self.grid = pv.UnstructuredGrid(cells, np.full(mesh.n_hexes, pv.CellType.HEXAHEDRON),
                                        pts)
        c = self.grid.cell_centers().points
        self._cell_theta = np.degrees(np.arctan2(c[:, 1], c[:, 2])) % 360.0
        self.R_out = R + mesh.t * thickness_scale

    def surface(self, T_nodes_K: np.ndarray, cutaway: str = "none",
                through_wall_mean: bool = False) -> pv.PolyData:
        """Visible wall surface with point data ``T_C``; with ``through_wall_mean`` every
        node shows the mean of its radial line (column)."""
        T = np.asarray(T_nodes_K, float)
        if through_wall_mean:
            cols = self.mesh.columns(T)
            w = self.mesh.r / self.mesh.r.sum()
            T = np.repeat(cols @ w, self.mesh.nr)
        self.grid.point_data["T_C"] = T - 273.15
        cut = CUTAWAYS[cutaway]
        solid = self.grid if cut is None else self.grid.extract_cells(
            np.flatnonzero(~((self._cell_theta > cut[0]) & (self._cell_theta < cut[1]))))
        return _surface(solid)
