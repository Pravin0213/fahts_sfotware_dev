"""
Phase 1.10 / 4.2 / 4.9 — colormap.py
Colour-mapping utilities for the FAHTS 3-D scene.

GroupColourMap
    Manages a discrete per-group colour assignment for beam elements.
    Writes 'group_id' integer scalars onto a PyVista mesh and produces
    a ListedColormap suitable for pv.Plotter.add_mesh().

TemperatureColourMap
    Wraps a named continuous colourmap (inferno, jet, plasma, coolwarm, hot)
    with a stored colour-limit range.  Writes 'temperature_C' scalars onto
    a PyVista mesh and provides add_mesh() keyword arguments for rendering.
    Supports an optional critical-temperature threshold overlay (Task 4.9)
    that replaces colours above T_crit with a vivid red alarm colour.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

# ── Threshold overlay constants ───────────────────────────────────────────────

#: Critical temperature for structural steel (EN 1993-1-2 §4.2.4) [°C].
CRITICAL_TEMPERATURE_STEEL: float = 660.0

#: RGBA colour used for the critical-temperature alarm overlay (vivid red).
THRESHOLD_COLOUR: tuple[float, float, float, float] = (1.0, 0.09, 0.27, 1.0)

# ── Default palette (RGB in [0, 1]) ───────────────────────────────────────────

DEFAULT_PALETTE: list[tuple[float, float, float]] = [
    (0.27, 0.51, 0.71),   # steel blue
    (0.87, 0.52, 0.32),   # burnt orange
    (0.40, 0.76, 0.65),   # teal
    (0.90, 0.72, 0.30),   # golden
    (0.70, 0.43, 0.68),   # purple
    (0.40, 0.68, 0.35),   # green
    (0.95, 0.50, 0.50),   # coral
    (0.60, 0.40, 0.25),   # brown
]

UNASSIGNED_COLOUR: tuple[float, float, float] = (0.56, 0.57, 0.59)   # steel grey


class GroupColourMap:
    """
    Discrete colour map for a fixed set of named element groups.

    Assigns one colour from *DEFAULT_PALETTE* to each group (cycling for
    more than 8 groups).  Individual colours can be overridden at
    construction time or later via :meth:`update`.

    The integer scalar written to the mesh is the index of the group in
    the sorted *group_names* list, or ``len(group_names)`` (the "sentinel"
    value) for elements that do not belong to any group — those cells
    receive *UNASSIGNED_COLOUR*.

    Parameters
    ----------
    group_names : list[str]
        Ordered list of group names (should be ``sorted(model.groups)``).
    overrides : dict[str, (r, g, b)], optional
        Explicit colour overrides; values not listed here get palette colours.
    """

    def __init__(
        self,
        group_names: list[str],
        overrides: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self._group_names: list[str] = list(group_names)
        self._colours: dict[str, tuple[float, float, float]] = {
            name: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
            for i, name in enumerate(self._group_names)
        }
        if overrides:
            self._colours.update(overrides)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def group_names(self) -> list[str]:
        """Ordered list of group names this map was built for."""
        return self._group_names

    @property
    def colours(self) -> dict[str, tuple[float, float, float]]:
        """Mapping of group name → (r, g, b) in [0, 1]."""
        return dict(self._colours)

    def colour_for(self, group_name: str) -> tuple[float, float, float]:
        """Return the colour for *group_name*, or *UNASSIGNED_COLOUR* if unknown."""
        return self._colours.get(group_name, UNASSIGNED_COLOUR)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def update(self, overrides: dict[str, tuple[float, float, float]]) -> None:
        """Override individual group colours in-place."""
        self._colours.update(overrides)

    # ── Colourmap construction ────────────────────────────────────────────────

    def build_cmap(self) -> ListedColormap:
        """
        Build a discrete matplotlib ListedColormap for this group set.

        The returned colormap has ``len(group_names) + 1`` entries: one per
        group in order, plus *UNASSIGNED_COLOUR* at the final index.
        """
        entries = [
            self._colours.get(name, UNASSIGNED_COLOUR)
            for name in self._group_names
        ]
        entries.append(UNASSIGNED_COLOUR)   # sentinel index = n_groups
        return ListedColormap(entries)

    # ── Mesh scalar writing ───────────────────────────────────────────────────

    def write_scalars(
        self,
        mesh: pv.PolyData,
        elem_to_group: dict[int, str],
    ) -> None:
        """
        Write a ``group_id`` integer cell-data array to *mesh* in-place.

        Parameters
        ----------
        mesh : pv.PolyData
            Must have an ``element_id`` cell-data array (written by
            :func:`~fahts.renderer.beam_geometry.build_model_mesh`).
        elem_to_group : dict[int, str]
            Maps element ID → group name.  Elements absent from this dict
            receive the sentinel value (``len(group_names)``).
        """
        n_groups = len(self._group_names)
        name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._group_names)
        }
        eid_arr = mesh.cell_data["element_id"]
        gid = np.array(
            [
                name_to_idx.get(elem_to_group.get(int(eid), ""), n_groups)
                for eid in eid_arr
            ],
            dtype=np.int32,
        )
        mesh.cell_data["group_id"] = gid

    # ── PyVista kwargs ────────────────────────────────────────────────────────

    def add_mesh_kwargs(self) -> dict[str, Any]:
        """
        Return keyword arguments for ``pv.Plotter.add_mesh()`` that will
        render the mesh coloured by group using this colour map.
        """
        n = len(self._group_names) + 1   # +1 for the unassigned sentinel
        return dict(
            scalars="group_id",
            cmap=self.build_cmap(),
            clim=(-0.5, n - 0.5),
            n_colors=n,
            show_scalar_bar=False,
            show_edges=False,
        )


# ── TemperatureColourMap ───────────────────────────────────────────────────────


class TemperatureColourMap:
    """
    Continuous temperature-to-colour mapping for beam elements.

    Wraps a named matplotlib / PyVista colourmap (e.g. ``'inferno'``,
    ``'jet'``) with a stored colour-limit range.  Provides helpers to write
    temperature scalars onto a PyVista mesh and to produce keyword arguments
    for ``pv.Plotter.add_mesh()``.

    Parameters
    ----------
    cmap : str
        Colourmap name.  Must be one of :attr:`SUPPORTED_CMAPS`.
    clim : (T_min, T_max) or None
        Initial colour limits [°C].  ``None`` until set by :meth:`auto_clim`
        or :meth:`set_clim`.
    n_colors : int
        Number of discrete colours in the LUT (default 256).
    """

    SUPPORTED_CMAPS: tuple[str, ...] = ("inferno", "jet", "plasma", "coolwarm", "hot")

    def __init__(
        self,
        cmap: str = "inferno",
        clim: tuple[float, float] | None = None,
        n_colors: int = 256,
        threshold_overlay: bool = False,
        T_crit: float = CRITICAL_TEMPERATURE_STEEL,
    ) -> None:
        if cmap not in self.SUPPORTED_CMAPS:
            raise ValueError(
                f"Unknown colormap {cmap!r}; supported: {self.SUPPORTED_CMAPS}"
            )
        self._cmap: str = cmap
        self._clim: tuple[float, float] | None = clim
        self._n_colors: int = n_colors
        self._threshold_overlay: bool = threshold_overlay
        self._T_crit: float = float(T_crit)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def cmap(self) -> str:
        """Current colourmap name."""
        return self._cmap

    @property
    def clim(self) -> tuple[float, float] | None:
        """Colour limits (T_min, T_max) [°C], or None if not yet set."""
        return self._clim

    @property
    def n_colors(self) -> int:
        """Number of discrete colours in the LUT."""
        return self._n_colors

    @property
    def threshold_overlay(self) -> bool:
        """True when the critical-temperature threshold overlay is active."""
        return self._threshold_overlay

    @property
    def T_crit(self) -> float:
        """Critical temperature threshold [°C] used by the overlay."""
        return self._T_crit

    # ── Mutation ──────────────────────────────────────────────────────────────

    def set_cmap(self, cmap: str) -> None:
        """Switch to a different colourmap by name."""
        if cmap not in self.SUPPORTED_CMAPS:
            raise ValueError(
                f"Unknown colormap {cmap!r}; supported: {self.SUPPORTED_CMAPS}"
            )
        self._cmap = cmap

    def set_clim(self, lo: float, hi: float) -> None:
        """Set colour limits explicitly [°C]."""
        self._clim = (float(lo), float(hi))

    def clear_clim(self) -> None:
        """Reset colour limits to None (unset)."""
        self._clim = None

    def set_threshold_overlay(
        self,
        enabled: bool,
        T_crit: float | None = None,
    ) -> None:
        """
        Enable or disable the critical-temperature threshold overlay.

        When enabled, :meth:`add_mesh_kwargs` returns a modified LUT where
        all colours representing temperatures at or above *T_crit* are
        replaced with :data:`THRESHOLD_COLOUR` (vivid red).

        Parameters
        ----------
        enabled : bool
            ``True`` to activate the overlay; ``False`` to revert to the
            plain continuous colourmap.
        T_crit : float or None
            New threshold temperature [°C].  ``None`` leaves the current
            threshold unchanged.
        """
        self._threshold_overlay = bool(enabled)
        if T_crit is not None:
            self._T_crit = float(T_crit)

    # ── Automatic colour limits ───────────────────────────────────────────────

    def auto_clim(
        self,
        T_values: Sequence[float] | np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute and store colour limits from *T_values*.

        A 1 °C pad is added when the range is less than 1 °C so the
        colourbar always has a non-zero extent.

        Returns
        -------
        tuple[float, float]
            The computed ``(lo, hi)`` pair.
        """
        arr = np.asarray(T_values, dtype=float)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        self._clim = (lo, hi + 1.0) if hi - lo < 1.0 else (lo, hi)
        return self._clim

    # ── Mesh scalar writing ───────────────────────────────────────────────────

    def write_scalars(
        self,
        mesh: pv.PolyData,
        T_per_element: dict[int, float],
        *,
        default_T: float = 20.0,
    ) -> np.ndarray:
        """
        Write ``temperature_C`` cell data (and derived point data) onto *mesh*.

        Cell values come from *T_per_element*; elements absent from that dict
        get *default_T*.  Point data is derived by averaging adjacent-cell
        values, giving smooth interpolation across beam joints.

        Parameters
        ----------
        mesh : pv.PolyData
            Must have an ``'element_id'`` cell-data array.
        T_per_element : dict[int, float]
            Maps element ID → temperature [°C].
        default_T : float
            Temperature for elements absent from *T_per_element*.

        Returns
        -------
        np.ndarray
            The 1-D cell temperature array that was written (float64).
        """
        eid_arr = mesh.cell_data["element_id"]
        T_arr = np.array(
            [T_per_element.get(int(eid), default_T) for eid in eid_arr],
            dtype=np.float64,
        )
        mesh.cell_data["temperature_C"] = T_arr
        try:
            tmp = mesh.cell_data_to_point_data()
            if "temperature_C" in tmp.point_data:
                mesh.point_data["temperature_C"] = tmp.point_data["temperature_C"]
        except Exception:  # noqa: BLE001 — non-fatal; cell data alone is sufficient
            pass
        return T_arr

    # ── Threshold LUT builder ─────────────────────────────────────────────────

    def _build_threshold_lut(self) -> ListedColormap:
        """
        Build a :class:`~matplotlib.colors.ListedColormap` from the current
        colourmap where all entries at or above *T_crit* are replaced with
        :data:`THRESHOLD_COLOUR`.

        Assumes ``self._clim`` is already set (caller's responsibility).
        """
        import matplotlib
        base = matplotlib.colormaps[self._cmap]
        lo, hi = self._clim  # type: ignore[misc]  # guarded by caller
        n = self._n_colors

        positions = np.linspace(0.0, 1.0, n)
        colours = base(positions)  # (n, 4) RGBA float64

        # Fraction along the [lo, hi] range where T_crit sits
        frac = (self._T_crit - lo) / (hi - lo) if hi > lo else 1.0
        frac = max(0.0, min(1.0, frac))

        if frac < 1.0:
            colours[positions >= frac] = THRESHOLD_COLOUR

        return ListedColormap(colours)

    # ── PyVista kwargs ────────────────────────────────────────────────────────

    def add_mesh_kwargs(self) -> dict[str, Any]:
        """
        Return keyword arguments for ``pv.Plotter.add_mesh()`` that render
        the mesh coloured by ``'temperature_C'`` using this colourmap.

        Raises
        ------
        ValueError
            If colour limits have not been set yet (call :meth:`auto_clim`
            or :meth:`set_clim` first).
        """
        if self._clim is None:
            raise ValueError(
                "Colour limits not set; call auto_clim() or set_clim() first."
            )
        if self._threshold_overlay:
            cmap: str | ListedColormap = self._build_threshold_lut()
        else:
            cmap = self._cmap

        title = (
            f"Temperature [°C]  (⚠ ≥{self._T_crit:.0f} °C)"
            if self._threshold_overlay
            else "Temperature [°C]"
        )
        scalar_bar_args: dict[str, Any] = {
            "title": title,
            "n_labels": 6,
            "title_font_size": 14,
            "label_font_size": 12,
            "color": "white",
            "vertical": True,
            "shadow": True,
        }

        return dict(
            scalars="temperature_C",
            cmap=cmap,
            clim=list(self._clim),
            n_colors=self._n_colors,
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_args,
            show_edges=False,
        )
