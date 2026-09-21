"""
Phase 1.5 — scene_manager.py
Owns and drives the PyVista plotter for the FAHTS 3-D scene.

Render modes
------------
  'wire'    — centreline line mesh only (fast overview)
  'section' — extruded BOX mesh at true scale (default)

Colour modes
------------
  'default'     — uniform steel grey
  'group'       — one colour per named group (discrete LUT)
  'temperature' — continuous inferno LUT mapped to °C values
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyvista as pv

from fahts.core.model.fem_model import FEMModel
from fahts.renderer.beam_geometry import build_centreline_mesh, build_model_mesh
from fahts.renderer.colormap import GroupColourMap, TemperatureColourMap, UNASSIGNED_COLOUR

log = logging.getLogger(__name__)

_STEEL_GREY       = UNASSIGNED_COLOUR        # (0.56, 0.57, 0.59) — re-exported alias
_STRUCTURE_COLOUR = (1.0, 0.95, 0.0)        # yellow default for structure
_FIRE_ORANGE      = (0.95, 0.45, 0.10)
_BACKGROUND       = (0.15, 0.15, 0.17)      # dark charcoal (like USFOS)
_TIME_LABEL_NAME  = "fahts_time_label"      # stable actor name for the time overlay


class SceneManager:
    """
    Manages a PyVista Plotter for the FAHTS 3-D scene.

    Parameters
    ----------
    plotter : pv.Plotter, optional
        Provide an existing plotter (e.g. pyvistaqt.BackgroundPlotter when
        embedded in a PyQt6 window).  If None, a new Plotter is created.
    off_screen : bool
        Only used when *plotter* is None.  Enables headless rendering for
        tests and batch export.
    """

    def __init__(
        self,
        plotter: pv.Plotter | None = None,
        *,
        off_screen: bool = False,
    ) -> None:
        if plotter is not None:
            self._pl = plotter
            self._owns_plotter = False
        else:
            self._pl = pv.Plotter(off_screen=off_screen)
            self._owns_plotter = True

        self._pl.set_background(color=_BACKGROUND)

        # Model state
        self._model: FEMModel | None = None
        self._centroid: np.ndarray = np.zeros(3)
        self._full_solid_mesh: pv.PolyData | None = None   # complete mesh; never modified
        self._solid_mesh: pv.PolyData | None = None        # current visible subset
        self._wire_mesh:  pv.PolyData | None = None

        # Actors — None when not currently added to the plotter
        self._solid_actor: Any = None
        self._wire_actor:  Any = None
        self._nodal_actor: Any = None           # per-vertex temperature mesh
        self._fire_zone_actors: list[Any] = []
        self._rad_ball_actors: list[Any] = []
        self._analysis_mesh_actor: Any = None   # FEM mesh wireframe overlay

        # Pipe geometry sides — kept in sync with c_circ from the mesh config
        self._n_pipe_sides: int = 16

        # Mode state
        self._render_mode: str = "section"   # 'wire' | 'section'
        self._colour_mode: str = "default"   # 'default' | 'group' | 'temperature'
        self._T_colour_map: TemperatureColourMap = TemperatureColourMap()
        self._T_field: Any = None   # TemperatureField; kept so visibility changes can reapply it
        self._T_t_idx: int = 0      # time-step index last applied
        self._nodal_mesh: Any = None  # pv.PolyData built from solver surface mesh nodes

        # Time label actor — text overlay showing t = X s when temperature is active
        self._time_label_actor: Any = None

        # Group colouring + visibility state
        self._group_names:      list[str] = []
        self._group_colour_map: GroupColourMap | None = None
        self._elem_to_group:    dict[int, str] = {}   # eid → group name
        self._group_visible:    dict[str, bool] = {}  # group name → visible flag

        # Custom legend clim — when set, overrides auto-range in update_temperature
        self._custom_clim: tuple[float, float] | None = None

        # Picking state
        self._pick_callback: Any = None
        self._vtk_pick_obs_id: Any = None   # VTK LeftButtonPressEvent observer id
        self._vtk_iren_ref: Any = None       # vtkRenderWindowInteractor ref (weak)

        # Mesh inspector state
        self._inspector_data: Any = None       # MeshInspectorData (when active, replaces solid render)
        self._inspector_callback: Any = None   # callable(beam_eid, quad_idx)

        # Highlight actor — element or inspector-quad highlight overlay
        self._highlight_actor: Any = None
        # True when highlight_element split _solid_mesh into rest+blue actors.
        self._highlight_split_active: bool = False
        # Node-index labels shown on the selected inspector quad
        self._inspector_node_label_actor: Any = None

        # USFOS-style floating element/node labels (Ctrl+click)
        self._element_label_actors: list[Any] = []

        # Axis marker state (world coordinates)
        self._axis_origin_world: np.ndarray = np.zeros(3)
        self._axis_marker_actors: list[Any] = []
        self._axis_marker_visible: bool = False

        # Coordinate-pick state (one-shot picking for fire-zone centre placement)
        self._coord_pick_callback: Any = None
        self._coord_pick_once: bool = True

        # External callback for live coordinate display in the status bar
        # Signature: callback(world_center: np.ndarray)
        self._coord_display_callback: Any = None

        # Scene tick timer — fires at 30 Hz independently of VTK render events.
        # Keeps axis actors anchored to the camera focal point (= screen centre)
        # and updates the status-bar coordinate display every tick.
        # Using a QTimer rather than add_on_render_callback because PyVista's
        # render callback uses a VTK interactor observer (StartEvent/EndEvent)
        # which fires on interaction start/end, NOT before every frame.
        self._scene_timer: Any = None
        try:
            from PyQt6.QtCore import QTimer as _QTimer
            self._scene_timer = _QTimer()
            self._scene_timer.setInterval(1)    # ~1 ms polling
            self._scene_timer.timeout.connect(self._on_scene_tick)
            self._scene_timer.start()
        except Exception:  # noqa: BLE001 — no Qt (headless test env)
            pass

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def plotter(self) -> pv.Plotter:
        """The underlying PyVista Plotter (use for embedding in Qt)."""
        return self._pl

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def colour_mode(self) -> str:
        return self._colour_mode

    @property
    def _T_clim(self) -> tuple[float, float] | None:
        """Colour limits currently active on _T_colour_map (read-only alias)."""
        return self._T_colour_map.clim

    @property
    def temperature_cmap(self) -> str:
        """Name of the current temperature colourmap."""
        return self._T_colour_map.cmap

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, model: FEMModel) -> None:
        """
        Build 3-D geometry from *model* and populate the scene.

        Any previously loaded model and fire-zone actors are removed first.
        Camera is reset to fit the new model.
        """
        self._clear_model_actors()
        self.clear_fire_zones()

        self._model = model
        self._centroid = model.centroid()
        self._colour_mode = "default"
        self._T_colour_map.clear_clim()
        self._T_field = None
        self._T_t_idx = 0
        self._nodal_mesh = None
        # Reset axis origin to model centroid so scene-coord origin = axis position
        self._axis_origin_world = self._centroid.copy()
        if self._axis_marker_visible:
            self._rebuild_axis_marker()

        log.info("Building solid mesh (%d elements)…", model.n_elements)
        self._full_solid_mesh = build_model_mesh(model, n_pipe_sides=self._n_pipe_sides)
        self._solid_mesh = self._full_solid_mesh   # all groups visible initially

        log.info("Building centreline mesh…")
        self._wire_mesh = build_centreline_mesh(model)

        self._build_group_mapping()        # also initialises _group_visible
        self._write_group_scalars()        # writes to _full_solid_mesh

        self._refresh_actors()
        self.reset_camera()
        log.info("Model loaded — %s", model.summary())

    # ── Render mode ───────────────────────────────────────────────────────────

    def set_render_mode(self, mode: str) -> None:
        """
        Switch between render modes without reloading geometry.

        Parameters
        ----------
        mode : 'wire' | 'section'
        """
        if mode not in ("wire", "section"):
            raise ValueError(f"Unknown render mode {mode!r}; expected 'wire' or 'section'")
        if mode == self._render_mode:
            return
        self._render_mode = mode
        self._refresh_actors()

    def set_pipe_sides(self, n: int) -> None:
        """
        Rebuild pipe geometry with *n* polygon sides so the solid mesh matches
        the circumferential mesh density chosen in the analysis config.

        No-op if *n* equals the current side count or no model is loaded.
        """
        if n < 3:
            raise ValueError(f"n_pipe_sides must be >= 3, got {n}")
        if n == self._n_pipe_sides or self._model is None:
            return
        self._n_pipe_sides = n
        self._full_solid_mesh = build_model_mesh(self._model, n_pipe_sides=n)
        self._solid_mesh = self._full_solid_mesh
        self._write_group_scalars()
        self._apply_visibility_filter()

    def show_analysis_mesh_overlay(self, mesh: "pv.PolyData | None") -> None:
        """
        Show or hide the FEM analysis mesh.

        Replaces the structural solid with the FEM mesh rendered as solid+edges
        so the cell boundaries are always aligned with the rendered surface.
        Pass the PolyData from build_analysis_mesh_overlay() to activate;
        pass None to remove and restore the structural mesh.
        """
        if self._analysis_mesh_actor is not None:
            try:
                self._pl.remove_actor(self._analysis_mesh_actor)
            except Exception:  # noqa: BLE001
                pass
            self._analysis_mesh_actor = None
            # Restore structural mesh
            self._refresh_actors()

        if mesh is not None and mesh.n_cells > 0:
            # Remove structural solid so there's no z-fighting between two surfaces
            self._remove_solid_actor()
            self._analysis_mesh_actor = self._pl.add_mesh(
                mesh,
                color=_STRUCTURE_COLOUR,
                show_edges=True,
                edge_color=_BACKGROUND,
                line_width=0.8,
                show_scalar_bar=False,
            )
        self._pl.render()

    # ── Mesh Inspector ────────────────────────────────────────────────────────

    def show_mesh_inspector(self, data: Any, callback: Any) -> None:
        """
        Activate the FEM mesh inspector.

        The structural mesh is replaced by the FEM surface mesh rendered as a
        solid surface with edges, so pick-target and visual edges are the same
        object — no misalignment.  Click any quad to fire *callback*.

        Parameters
        ----------
        data     : MeshInspectorData from build_mesh_inspector_data().
        callback : callable(beam_eid: int, quad_idx: int) — fired when a quad is clicked.
        """
        self.hide_mesh_inspector()
        self._inspector_data = data
        self._inspector_callback = callback
        self._refresh_actors()   # swaps structural → FEM mesh rendering

    def hide_mesh_inspector(self) -> None:
        """Deactivate the inspector and restore structural mesh rendering."""
        self._inspector_data = None
        self._inspector_callback = None
        self._remove_highlight()
        self._refresh_actors()

    # ── Highlight ─────────────────────────────────────────────────────────────

    def highlight_element(self, eid: int) -> None:
        """
        Highlight all faces belonging to *eid* in solid blue.

        The solid mesh is split into two disjoint subsets: the selected
        element (blue) and everything else (yellow).  Because neither subset
        is co-planar with the other, there is zero z-fighting and no edge
        lines bleed through.  All actor add/remove calls use render=False so
        only a single composite frame is drawn at the end — no blink.
        """
        self._remove_highlight()   # render=False internally

        if self._inspector_data is not None and self._inspector_data.mesh.n_cells > 0:
            target = self._inspector_data.mesh  # type: ignore[union-attr]
            key = "inspector_beam_eid"
            inspector_mode = True
        elif self._solid_mesh is not None and self._solid_mesh.n_cells > 0:
            target = self._solid_mesh
            key = "element_id"
            inspector_mode = False
        else:
            return

        eid_arr = np.asarray(target.cell_data.get(key, []))
        sel_idx = np.where(eid_arr == eid)[0]
        if len(sel_idx) == 0:
            return

        sel_mesh = target.extract_cells(sel_idx)

        if inspector_mode:
            # Inspector mesh is already a separate object — simple overlay with
            # render=False so the single render at the end is the only frame.
            try:
                fill_actor = self._pl.add_mesh(
                    sel_mesh,
                    color=(0.15, 0.45, 1.0),
                    lighting=False,
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=False,
                    pickable=False,
                    render=False,
                )
            except TypeError:   # older PyVista without render kwarg
                fill_actor = self._pl.add_mesh(
                    sel_mesh,
                    color=(0.15, 0.45, 1.0),
                    lighting=False,
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=False,
                    pickable=False,
                )
            self._highlight_actor = fill_actor
        else:
            # Split the solid mesh — selected element gets its own blue actor,
            # rest goes into a separate yellow actor.  No co-planar surfaces,
            # no polygon-offset tricks needed, no GL_LINE bleed-through.
            rest_idx = np.where(eid_arr != eid)[0]
            # Remove existing solid actor (render=False — no intermediate frame)
            self._remove_solid_actor()
            if len(rest_idx) > 0:
                rest_mesh = target.extract_cells(rest_idx)
                try:
                    self._solid_actor = self._pl.add_mesh(
                        rest_mesh, render=False, **self._solid_mesh_kwargs()
                    )
                except TypeError:
                    self._solid_actor = self._pl.add_mesh(
                        rest_mesh, **self._solid_mesh_kwargs()
                    )
            try:
                blue_actor = self._pl.add_mesh(
                    sel_mesh,
                    color=(0.15, 0.45, 1.0),
                    show_edges=True,
                    edge_color=_BACKGROUND,
                    line_width=0.8,
                    show_scalar_bar=False,
                    pickable=False,
                    render=False,
                )
            except TypeError:
                blue_actor = self._pl.add_mesh(
                    sel_mesh,
                    color=(0.15, 0.45, 1.0),
                    show_edges=True,
                    edge_color=_BACKGROUND,
                    line_width=0.8,
                    show_scalar_bar=False,
                    pickable=False,
                )
            self._highlight_actor = blue_actor
            self._highlight_split_active = True

        self._pl.render()   # single composite frame

    def highlight_inspector_quad(self, beam_eid: int, quad_idx: int) -> None:
        """
        Highlight a single FEM quad (used by the mesh inspector on click).

        The quad is shown with an orange-red fill so it stands out from the
        rest of the FEM mesh.
        """
        self._remove_highlight()
        if self._inspector_data is None:
            return
        offset = self._inspector_data.beam_cell_offset.get(beam_eid)
        if offset is None:
            return
        cell_idx = offset + quad_idx
        if cell_idx >= self._inspector_data.mesh.n_cells:
            return
        highlighted = self._inspector_data.mesh.extract_cells([cell_idx])
        fill_actor = self._pl.add_mesh(
            highlighted,
            color=(1.0, 0.45, 0.0),
            opacity=1.0,
            show_edges=True,
            edge_color=(1.0, 1.0, 1.0),
            line_width=1.5,
            show_scalar_bar=False,
            pickable=False,
        )
        try:
            prop = fill_actor.GetProperty()
            prop.PolygonOffsetOn()
            prop.SetPolygonOffsetFactor(-2.0)
            prop.SetPolygonOffsetUnits(-2.0)
        except Exception:  # noqa: BLE001
            pass
        self._highlight_actor = fill_actor

        # Render global-DOF labels at each corner of the selected quad.
        # Use global mesh point lookup via beam_node_offset — extract_cells reorders
        # points by ascending global index, so highlighted.points cannot be used.
        quad_nodes = self._inspector_data.beam_quads.get(beam_eid)
        node_offset = self._inspector_data.beam_node_offset.get(beam_eid, 0)
        gdof_arr   = self._inspector_data.beam_node_gdof.get(beam_eid)
        if quad_nodes is not None and quad_idx < len(quad_nodes):
            node_indices = quad_nodes[quad_idx]   # 4 local node indices
            pts = np.array([
                self._inspector_data.mesh.points[int(n) + node_offset]
                for n in node_indices
            ])
            if gdof_arr is not None:
                labels = [str(int(gdof_arr[int(n)])) for n in node_indices]
            else:
                labels = [str(int(n)) for n in node_indices]
            try:
                self._inspector_node_label_actor = self._pl.add_point_labels(
                    pts,
                    labels,
                    font_size=12,
                    text_color=(1.0, 1.0, 1.0),
                    font_family="courier",
                    bold=True,
                    show_points=False,
                    always_visible=True,
                    reset_camera=False,
                )
            except Exception:  # noqa: BLE001
                pass

        self._pl.render()

    def _remove_highlight(self) -> None:
        """
        Remove highlight actor(s) without triggering an intermediate render.

        When the split-highlight is active the solid actor holds the rest-mesh;
        removing it here lets _refresh_actors (or the caller) rebuild the full
        solid mesh cleanly without leaving a gap frame.
        """
        if self._highlight_actor is not None:
            actors = (
                self._highlight_actor
                if isinstance(self._highlight_actor, tuple)
                else (self._highlight_actor,)
            )
            for actor in actors:
                try:
                    self._pl.remove_actor(actor, render=False)
                except Exception:  # noqa: BLE001
                    pass
            self._highlight_actor = None

        if self._highlight_split_active:
            self._remove_solid_actor()   # removes rest-mesh (render=False)
            self._highlight_split_active = False

        if self._inspector_node_label_actor is not None:
            try:
                self._pl.remove_actor(self._inspector_node_label_actor, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._inspector_node_label_actor = None

    def _clear_element_labels(self) -> None:
        """Remove USFOS-style floating element/node labels from the scene."""
        for actor in self._element_label_actors:
            try:
                self._pl.remove_actor(actor, render=False)
            except Exception:  # noqa: BLE001
                pass
        self._element_label_actors.clear()

    def _show_element_labels(self, eid: int, pick_center: np.ndarray) -> None:
        """
        Show USFOS-style floating labels for a Ctrl+click selected element.

        Handles both beam elements (n1/n2 endpoints → "element end 1/2") and
        shell elements (corner node tuple → nearest corner node).

        Two labels rendered in scene space:
          • "Element {eid}"       — near the element centroid
          • "Node {nid}, …"       — near the nearest node to the click position
        """
        self._clear_element_labels()
        if self._model is None:
            return

        # ── Beam element ──────────────────────────────────────────────────────
        beam = self._model.elements.get(eid)
        if beam is not None:
            n1 = self._model.nodes.get(beam.n1)
            n2 = self._model.nodes.get(beam.n2)
            if n1 is None or n2 is None:
                return
            p1 = n1.xyz - self._centroid
            p2 = n2.xyz - self._centroid
            elem_center = (p1 + p2) * 0.5
            if np.linalg.norm(pick_center - p1) <= np.linalg.norm(pick_center - p2):
                nearest_nid, node_pos, node_label = beam.n1, p1, f"Node {beam.n1}, element end 1"
            else:
                nearest_nid, node_pos, node_label = beam.n2, p2, f"Node {beam.n2}, element end 2"

        # ── Shell element ─────────────────────────────────────────────────────
        else:
            shell = self._model.shell_elements.get(eid)
            if shell is None:
                return
            # Collect scene-space positions for all corner nodes
            corner_pts: list[tuple[int, np.ndarray]] = []
            for nid in shell.nodes:
                node = self._model.nodes.get(nid)
                if node is not None:
                    corner_pts.append((nid, node.xyz - self._centroid))
            if not corner_pts:
                return
            pts = np.array([p for _, p in corner_pts])
            elem_center = pts.mean(axis=0)
            # Nearest corner node to the pick position
            dists = [np.linalg.norm(pick_center - p) for _, p in corner_pts]
            nearest_nid, node_pos = corner_pts[int(np.argmin(dists))]
            node_label = f"Node {nearest_nid}"

        def _add_label(pos: np.ndarray, text: str, text_color: tuple) -> None:
            try:
                actor = self._pl.add_point_labels(
                    np.array([pos]),
                    [text],
                    font_size=12,
                    bold=False,
                    italic=False,
                    text_color=text_color,
                    shape_color=(1.0, 1.0, 1.0),
                    shape="rounded_rect",
                    show_points=False,
                    always_visible=True,
                    reset_camera=False,
                )
                self._element_label_actors.append(actor)
            except Exception:  # noqa: BLE001
                pass

        _add_label(elem_center, f"Element {eid}", (0.80, 0.08, 0.08))
        _add_label(node_pos, node_label, (0.08, 0.35, 0.85))
        self._pl.render()

    # ── Colouring ─────────────────────────────────────────────────────────────

    def colour_by_group(
        self,
        group_colours: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        """
        Colour beams by their named group using a discrete LUT.

        Parameters
        ----------
        group_colours : optional override map  {group_name → (r, g, b)}
            Values in [0, 1].  If None the built-in palette is used.
        """
        if self._solid_mesh is None or self._group_colour_map is None:
            return
        if group_colours:
            self._group_colour_map.update(group_colours)
        self._colour_mode = "group"
        self._refresh_actors()

    def colour_by_temperature(
        self,
        T_per_element: dict[int, float],
        *,
        clim: tuple[float, float] | None = None,
    ) -> None:
        """
        Colour beams by per-element temperature [°C].

        Parameters
        ----------
        T_per_element : {element_id: temperature [°C]}
            Elements not in the dict default to 20 °C.
        clim : (T_min, T_max) colour limits; auto-derived from data if None.
        """
        if self._solid_mesh is None:
            return

        T_arr = self._T_colour_map.write_scalars(self._solid_mesh, T_per_element)

        prev_clim = self._T_colour_map.clim
        if clim is not None:
            self._T_colour_map.set_clim(*clim)
        else:
            self._T_colour_map.auto_clim(T_arr)

        self._colour_mode = "temperature"

        # Fast path: actor already exists and colour limits haven't changed →
        # update scalars in-place without the blink-inducing remove/re-add cycle.
        if (self._solid_actor is not None
                and self._render_mode == "section"
                and self._T_colour_map.clim == prev_clim):
            self._solid_mesh.Modified()
            self._pl.render()
        else:
            self._refresh_actors()

    def set_temperature_cmap(self, cmap: str) -> None:
        """
        Switch the temperature colourmap by name.

        Parameters
        ----------
        cmap : str
            One of ``TemperatureColourMap.SUPPORTED_CMAPS``
            (``'inferno'``, ``'jet'``, ``'plasma'``, ``'coolwarm'``, ``'hot'``).

        If temperature data is currently displayed, the scene refreshes
        immediately.  Otherwise the change takes effect on the next call to
        :meth:`colour_by_temperature` or :meth:`update_temperature`.
        """
        self._T_colour_map.set_cmap(cmap)
        if self._colour_mode == "temperature":
            self._refresh_actors()

    def set_threshold_overlay(
        self,
        enabled: bool,
        T_crit: float | None = None,
    ) -> None:
        """
        Enable or disable the critical-temperature threshold overlay.

        Colours representing temperatures at or above *T_crit* are replaced
        with a vivid red alarm colour in the LUT.  The scene refreshes
        immediately when temperature data is currently displayed.

        Parameters
        ----------
        enabled : bool
            ``True`` to activate the overlay.
        T_crit : float or None
            Override the threshold [°C]; ``None`` keeps the current value.
        """
        self._T_colour_map.set_threshold_overlay(enabled, T_crit=T_crit)
        if self._colour_mode == "temperature":
            self._refresh_actors()

    @property
    def legend_clim(self) -> tuple[float, float] | None:
        """Custom legend clim if set by the user, otherwise None (= auto)."""
        return self._custom_clim

    def set_legend_clim(self, lo: float, hi: float) -> None:
        """
        Lock the legend fringe range to [lo, hi] °C.

        Persists across animation steps.  The scene refreshes immediately when
        temperature data is currently displayed.
        """
        self._custom_clim = (float(lo), float(hi))
        if self._colour_mode == "temperature" and self._T_field is not None:
            if getattr(self._T_field, "nodal_geometry", {}):
                self._T_colour_map.set_clim(*self._custom_clim)
                self._refresh_actors()
            else:
                T_row = self._T_field.T_centroid[self._T_t_idx]
                T_map = {eid: float(T_row[i]) for i, eid in enumerate(self._T_field.element_ids)}
                self.colour_by_temperature(T_map, clim=self._custom_clim)

    def reset_legend_clim(self) -> None:
        """
        Remove the custom clim lock and revert to auto-range derived from data.

        The scene refreshes immediately when temperature data is displayed.
        """
        self._custom_clim = None
        if self._colour_mode == "temperature" and self._T_field is not None:
            T_all = np.asarray(self._T_field.T_centroid)
            lo, hi = float(np.nanmin(T_all)), float(np.nanmax(T_all))
            clim = (lo, hi + 1.0) if hi - lo < 1.0 else (lo, hi)
            if getattr(self._T_field, "nodal_geometry", {}):
                self._T_colour_map.set_clim(*clim)
                self._refresh_actors()
            else:
                T_row = self._T_field.T_centroid[self._T_t_idx]
                T_map = {eid: float(T_row[i]) for i, eid in enumerate(self._T_field.element_ids)}
                self.colour_by_temperature(T_map, clim=clim)

    def reset_colour(self) -> None:
        """Return to the default uniform steel-grey colouring."""
        self._colour_mode = "default"
        self._nodal_mesh = None
        self._refresh_actors()

    # ── Group visibility ──────────────────────────────────────────────────────

    def set_group_visibility(self, group_name: str, visible: bool) -> None:
        """
        Show or hide all beams belonging to *group_name*.

        Parameters
        ----------
        group_name : str
            Must match a key in the loaded model's groups dict.
        visible : bool
        """
        if group_name not in self._group_visible:
            log.warning("set_group_visibility: unknown group %r", group_name)
            return
        self._group_visible[group_name] = visible
        self._apply_visibility_filter()

    def set_all_groups_visibility(self, visible: bool) -> None:
        """Show or hide all groups at once (single mesh rebuild)."""
        for name in self._group_visible:
            self._group_visible[name] = visible
        self._apply_visibility_filter()

    def get_group_visibility(self, group_name: str) -> bool:
        """Return the current visibility flag for *group_name* (True = visible)."""
        return self._group_visible.get(group_name, True)

    # ── Element picking ───────────────────────────────────────────────────────

    def enable_picking(self, callback: Any) -> None:
        """
        Enable Ctrl+left-click element picking on the solid beam mesh (USFOS style).

        Parameters
        ----------
        callback : callable(element_id: int)
            Called with the integer element ID of the clicked beam face.
            Only fires when Ctrl is held at the time of the click.
        """
        self._pick_callback = callback
        self._deregister_vtk_pick_observer()
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass
        # Register a LeftButtonPressEvent VTK observer.  This fires before the
        # interactor style handles the event, giving us reliable Ctrl detection
        # via interactor.GetControlKey() rather than Qt's event queue.
        # pyvistaqt stores the raw vtkRenderWindowInteractor at iren.interactor
        # (confirmed from pyvistaqt source: self.iren.interactor.RemoveObservers(...)).
        try:
            iren = self._pl.iren.interactor  # vtkRenderWindowInteractor
            obs_id = iren.AddObserver(
                "LeftButtonPressEvent", self._on_vtk_left_press, 0.5
            )
            self._vtk_pick_obs_id = obs_id
            self._vtk_iren_ref = iren
        except Exception:  # noqa: BLE001 — headless / no VTK interactor
            # Fallback: use PyVista's built-in cell picker (no Ctrl filtering)
            self._pl.enable_cell_picking(
                callback=self._on_cell_picked,
                through=False,
                show=False,
                show_message=False,
            )

    def disable_picking(self) -> None:
        """Disable element picking."""
        self._pick_callback = None
        self._deregister_vtk_pick_observer()
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass

    def _deregister_vtk_pick_observer(self) -> None:
        """Remove the LeftButtonPressEvent VTK observer if registered."""
        if self._vtk_pick_obs_id is not None and self._vtk_iren_ref is not None:
            try:
                self._vtk_iren_ref.RemoveObserver(self._vtk_pick_obs_id)
            except Exception:  # noqa: BLE001
                pass
        self._vtk_pick_obs_id = None
        self._vtk_iren_ref = None

    def _on_vtk_left_press(self, interactor: Any, event: str) -> None:
        """
        VTK LeftButtonPressEvent observer — USFOS-style Ctrl+click selection.

        Ctrl+click  → pick element, highlight, show floating labels.
        Plain click → pick inspector quad (if inspector active); otherwise ignored.
        """
        ctrl = bool(interactor.GetControlKey())
        x, y = interactor.GetEventPosition()

        # Import vtkCellPicker (works for both old vtk and new vtkmodules layouts)
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker
        except ImportError:
            try:
                from vtk import vtkCellPicker  # type: ignore[no-redef]
            except ImportError:
                return

        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, self._pl.renderer)
        cell_id = picker.GetCellId()

        if cell_id < 0:
            if ctrl:
                was_split = self._highlight_split_active
                self._clear_element_labels()
                self._remove_highlight()   # render=False internally
                if was_split:
                    self._refresh_actors() # rebuilds full solid mesh + renders
                else:
                    self._pl.render()
            return

        # Read cell arrays from the ACTUALLY picked VTK dataset so the element
        # id is always correct regardless of which actor was rendered last
        # (avoids wrong-element selection when the scene contains multiple actors).
        vtk_ds = picker.GetDataSet()
        if vtk_ds is None:
            return
        cda = vtk_ds.GetCellData()

        # Inspector plain click — look for inspector_beam_eid on the dataset
        if (
            not ctrl
            and self._inspector_callback is not None
            and self._inspector_data is not None
            and self._inspector_data.mesh.n_cells > 0
        ):
            insp_eid_vtk = cda.GetArray("inspector_beam_eid")
            insp_qidx_vtk = cda.GetArray("inspector_quad_idx")
            if insp_eid_vtk is not None and insp_qidx_vtk is not None:
                self._inspector_callback(
                    int(insp_eid_vtk.GetValue(cell_id)),
                    int(insp_qidx_vtk.GetValue(cell_id)),
                )
            return

        # Element Ctrl+click — look for element_id on the dataset
        if not ctrl or self._pick_callback is None:
            return

        eid_vtk = cda.GetArray("element_id")
        if eid_vtk is None:
            # Inspector mesh uses inspector_beam_eid; accept that too
            eid_vtk = cda.GetArray("inspector_beam_eid")
        if eid_vtk is None:
            return  # picked a fire zone, rad ball, or other non-structural actor

        eid = int(eid_vtk.GetValue(cell_id))
        pick_pos = np.array(picker.GetPickPosition(), dtype=float)
        self._show_element_labels(eid, pick_pos)
        self._pick_callback(eid)

    def _on_cell_picked(self, picked: Any) -> None:
        """
        Headless fallback for PyVista's enable_cell_picking (used when the VTK
        interactor is unavailable, e.g. in off-screen tests).

        Routes to the inspector callback when an inspector quad is clicked
        (cell data has ``inspector_beam_eid``), otherwise to the element
        callback (cell data has ``element_id``).
        """
        if picked is None:
            return
        cell_data = getattr(picked, "cell_data", None)
        if cell_data is None:
            return

        # Inspector quad
        if (
            "inspector_beam_eid" in cell_data
            and self._inspector_callback is not None
        ):
            beids = cell_data["inspector_beam_eid"]
            qidxs = cell_data["inspector_quad_idx"]
            if len(beids) > 0:
                self._inspector_callback(int(beids[0]), int(qidxs[0]))
            return

        # Element (beam/shell) pick
        if "element_id" in cell_data and self._pick_callback is not None:
            eids = cell_data["element_id"]
            if len(eids) > 0:
                eid = int(eids[0])
                try:
                    pick_center = np.asarray(picked.center, dtype=float)
                except Exception:  # noqa: BLE001
                    pick_center = np.zeros(3)
                self._show_element_labels(eid, pick_center)
                self._pick_callback(eid)

    # ── Fire zones ────────────────────────────────────────────────────────────

    def show_fire_zones(self, zones: Sequence[Any]) -> None:
        """
        Render each fire zone as a semi-transparent orange box.

        Each *zone* must expose:
            .center  — (x, y, z) global centre [m]
            .dims    — (dx, dy, dz) full extents [m]
            .name    — str  (used as actor label)

        Coordinates are translated by the stored model centroid so the boxes
        align with the centroid-shifted beam geometry.
        """
        self.clear_fire_zones()
        for zone in zones:
            cx, cy, cz = (float(v) for v in zone.center)
            dx, dy, dz = (float(v) for v in zone.dims)
            # Apply the same centroid shift used by build_model_mesh()
            cx -= self._centroid[0]
            cy -= self._centroid[1]
            cz -= self._centroid[2]

            box = pv.Box(bounds=(
                cx - dx / 2, cx + dx / 2,
                cy - dy / 2, cy + dy / 2,
                cz - dz / 2, cz + dz / 2,
            ))
            actor = self._pl.add_mesh(
                box,
                color=_FIRE_ORANGE,
                opacity=0.30,
                style="surface",
                show_edges=True,
                edge_color="darkorange",
            )
            self._fire_zone_actors.append(actor)

        self._pl.render()

    def clear_fire_zones(self) -> None:
        """Remove all fire-zone box actors from the scene."""
        for actor in self._fire_zone_actors:
            try:
                self._pl.remove_actor(actor)
            except Exception:  # noqa: BLE001
                pass
        self._fire_zone_actors.clear()
        self._pl.render()

    def show_rad_balls(self, balls: Sequence[Any]) -> None:
        """
        Render each radiation ball as a semi-transparent yellow sphere, with a
        thin wireframe overlay to make the boundary legible.

        Each *ball* must expose:
            .center  — (x, y, z) global centre [m]
            .radius  — ball radius [m]
            .name    — str
        """
        self.clear_rad_balls()
        for ball in balls:
            cx = float(ball.center[0]) - self._centroid[0]
            cy = float(ball.center[1]) - self._centroid[1]
            cz = float(ball.center[2]) - self._centroid[2]
            centre = (cx, cy, cz)

            sphere = pv.Sphere(radius=float(ball.radius), center=centre)

            # Solid semi-transparent surface
            a1 = self._pl.add_mesh(
                sphere, color=(1.0, 0.85, 0.0), opacity=0.20, style="surface",
            )
            self._rad_ball_actors.append(a1)

            # Wireframe overlay for a legible boundary
            a2 = self._pl.add_mesh(
                sphere, color=(1.0, 0.85, 0.0), opacity=0.35,
                style="wireframe", line_width=1,
            )
            self._rad_ball_actors.append(a2)

        self._pl.render()

    def clear_rad_balls(self) -> None:
        """Remove all radiation-ball sphere actors from the scene."""
        for actor in self._rad_ball_actors:
            try:
                self._pl.remove_actor(actor)
            except Exception:  # noqa: BLE001
                pass
        self._rad_ball_actors.clear()
        self._pl.render()

    def highlight_exposed_beams(self, exposed_eids: set[int]) -> None:
        """
        Colour exposed beams with a red-orange tint; unexposed beams stay grey.

        Parameters
        ----------
        exposed_eids : set of element IDs that are inside a fire zone.
            Pass an empty set to clear the highlight and revert to default colour.
        """
        if self._solid_mesh is None:
            return

        if not exposed_eids:
            self.reset_colour()
            return

        # Build a per-cell scalar: 1.0 = exposed, 0.0 = not exposed
        eid_arr = self._solid_mesh.cell_data["element_id"]
        exposed_arr = np.array(
            [1.0 if int(eid) in exposed_eids else 0.0 for eid in eid_arr],
            dtype=np.float64,
        )
        self._solid_mesh.cell_data["exposed"] = exposed_arr
        self._colour_mode = "exposed"
        self._refresh_actors()

    def _solid_mesh_kwargs_exposed(self) -> dict[str, Any]:
        """Kwargs for the exposed-beam highlight colouring."""
        return dict(
            scalars="exposed",
            cmap=["#8F9196", "#E8420A"],   # grey → fire-orange
            clim=[0.0, 1.0],
            show_scalar_bar=False,
            show_edges=False,
        )

    # ── Camera ────────────────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Fit the camera to the current scene bounding box."""
        self._pl.reset_camera()
        self._pl.render()

    def set_camera_view(self, view: str) -> None:
        """
        Snap to a standard orthographic direction.

        Parameters
        ----------
        view : '+x' | '-x' | '+y' | '-y' | '+z' | '-z'
        """
        _directions = {
            "+x": ((1, 0, 0), (0, 0, 1)),
            "-x": ((-1, 0, 0), (0, 0, 1)),
            "+y": ((0, 1, 0), (0, 0, 1)),
            "-y": ((0, -1, 0), (0, 0, 1)),
            "+z": ((0, 0, 1), (0, 1, 0)),
            "-z": ((0, 0, -1), (0, 1, 0)),
        }
        if view not in _directions:
            raise ValueError(f"Unknown view {view!r}")
        direction, viewup = _directions[view]
        self._pl.view_vector(direction, viewup=viewup)
        self._pl.reset_camera()
        self._pl.render()

    # ── Temperature animation ─────────────────────────────────────────────────

    def update_temperature(self, t: float, T_field: Any) -> None:
        """
        Snap the display to the time step in *T_field* closest to *t* [s].

        When *T_field* carries nodal geometry (populated by the surface solver),
        colours are applied per mesh vertex, showing within-element temperature
        gradients.  Otherwise falls back to per-element centroid colouring.

        *T_field* duck-type (matches TemperatureField dataclass from Phase 3):
            .times           np.ndarray  (n_steps,)
            .element_ids     list[int]
            .T_centroid      np.ndarray  (n_steps, n_elems)
            .nodal_geometry  dict        {eid: (nodes_global, quads)}  — optional
        """
        if T_field is None or self._solid_mesh is None:
            return

        times = np.asarray(T_field.times)
        idx = int(np.argmin(np.abs(times - t)))

        self._T_field = T_field
        self._T_t_idx = idx

        if self._custom_clim is not None:
            clim = self._custom_clim
        else:
            T_all = np.asarray(T_field.T_centroid)
            lo, hi = float(np.nanmin(T_all)), float(np.nanmax(T_all))
            clim = (lo, hi + 1.0) if hi - lo < 1.0 else (lo, hi)

        ng = getattr(T_field, "nodal_geometry", {})
        if ng:
            nodal_mesh = self._build_nodal_pv_mesh(T_field, idx)
            if nodal_mesh is not None:
                self._nodal_mesh = nodal_mesh
                self._T_colour_map.set_clim(*clim)
                self._colour_mode = "temperature"
                self._refresh_actors()
                self._update_time_label(float(t))
                return

        # Centroid fallback: one uniform colour per element
        T_row = T_field.T_centroid[idx]
        T_map = {eid: float(T_row[i]) for i, eid in enumerate(T_field.element_ids)}
        self.colour_by_temperature(T_map, clim=clim)
        self._update_time_label(float(t))

    def animate(
        self,
        T_field: Any,
        fps: int = 10,
        callback: Any = None,
    ) -> None:
        """
        Step through all time steps in *T_field*, updating the display each frame.

        Parameters
        ----------
        T_field : TemperatureField
        fps     : target frame rate (wall-clock throttle; no VSync guarantee)
        callback : optional callable(t: float, T_row: np.ndarray) per step
        """
        if T_field is None:
            return

        dt_target = 1.0 / max(fps, 1)
        times = np.asarray(T_field.times)

        for idx, t in enumerate(times):
            t0 = time.monotonic()
            self.update_temperature(float(t), T_field)
            if callback is not None:
                callback(float(t), T_field.T_centroid[idx])
            lag = dt_target - (time.monotonic() - t0)
            if lag > 0:
                time.sleep(lag)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def screenshot(self, path: str | Path) -> None:
        """Save a PNG screenshot of the current scene."""
        self._pl.screenshot(str(path))
        log.info("Screenshot saved → %s", path)

    def save_animation(
        self,
        T_field: object,
        path: str | Path,
        fps: int = 10,
        progress_callback: object = None,
    ) -> int:
        """
        Render a temperature animation to a GIF (or MP4 if imageio-ffmpeg installed).

        Parameters
        ----------
        T_field          : TemperatureField with .times array
        path             : Output path. Extension must be .gif, .mp4, or .avi.
        fps              : Frames per second for the output file.
        progress_callback: Optional callable(current_frame, total_frames).

        Returns
        -------
        Number of frames written.

        Raises
        ------
        ValueError      : Unsupported file extension.
        ImportError     : MP4/AVI requested but imageio-ffmpeg not installed.
        """
        import imageio  # noqa: PLC0415

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in {".gif", ".mp4", ".avi"}:
            raise ValueError(
                f"Unsupported animation format {suffix!r}. Use .gif, .mp4, or .avi."
            )

        if suffix in (".mp4", ".avi"):
            try:
                import imageio_ffmpeg  # noqa: F401, PLC0415
            except ImportError:
                raise ImportError(
                    f"Saving {suffix} requires imageio-ffmpeg: "
                    "pip install imageio-ffmpeg"
                ) from None

        times = np.asarray(T_field.times)
        n_steps = len(times)

        if suffix == ".gif":
            # imageio pillow plugin uses 'duration' (ms per frame), not 'fps'
            writer_kwargs: dict = {"duration": round(1000 / max(fps, 1)), "loop": 0}
        else:
            writer_kwargs = {"fps": fps}

        with imageio.get_writer(str(path), mode="I", **writer_kwargs) as writer:
            for step_idx, t in enumerate(times):
                self.update_temperature(float(t), T_field)
                frame = self._pl.screenshot(return_img=True)
                writer.append_data(frame)
                if progress_callback is not None:
                    progress_callback(step_idx + 1, n_steps)

        log.info(
            "Animation written → %s  (%d frames @ %d fps)", path, n_steps, fps
        )
        return n_steps

    def show(self) -> None:
        """Open the plotter window (blocks until closed; standalone use only)."""
        self._pl.show()

    def close(self) -> None:
        """Close and clean up the plotter (call on app exit)."""
        if self._scene_timer is not None:
            try:
                self._scene_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._owns_plotter:
            self._pl.close()

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_group_mapping(self) -> None:
        """Populate _group_names, _group_colour_map, _elem_to_group, _group_visible."""
        assert self._model is not None
        self._elem_to_group.clear()
        self._group_names = sorted(self._model.groups.keys())

        for name, group in self._model.groups.items():
            for eid in group.element_ids:
                self._elem_to_group[eid] = name   # last group wins if multiple

        self._group_colour_map = GroupColourMap(self._group_names)

        # All groups visible by default when a new model is loaded
        self._group_visible = {name: True for name in self._group_names}

    def _write_group_scalars(self) -> None:
        """Write integer 'group_id' cell array into self._full_solid_mesh."""
        if self._full_solid_mesh is None or self._group_colour_map is None:
            return
        self._group_colour_map.write_scalars(self._full_solid_mesh, self._elem_to_group)

    def _solid_mesh_kwargs(self) -> dict[str, Any]:
        """Return add_mesh kwargs for the current colour mode."""
        assert self._solid_mesh is not None

        if (
            self._colour_mode == "group"
            and self._group_colour_map is not None
            and self._group_names
        ):
            return self._group_colour_map.add_mesh_kwargs()

        if (
            self._colour_mode == "temperature"
            and "temperature_C" in self._solid_mesh.cell_data
            and self._T_colour_map.clim is not None
        ):
            return self._T_colour_map.add_mesh_kwargs()

        if (
            self._colour_mode == "exposed"
            and "exposed" in self._solid_mesh.cell_data
        ):
            return self._solid_mesh_kwargs_exposed()

        # default — show edges so element boundaries are visible (USFOS style)
        return dict(
            color=_STRUCTURE_COLOUR,
            show_scalar_bar=False,
            show_edges=True,
            edge_color=_BACKGROUND,
            line_width=0.8,
        )

    def _update_time_label(self, t: float | None) -> None:
        """
        Show, update, or remove the simulation-time text overlay.

        The label appears in the upper-left corner of the viewport when
        temperature data is active.  Passing ``None`` removes it.

        Format examples:
            ``t = 0 s``
            ``t = 3600 s  (60.0 min)``

        Parameters
        ----------
        t : float or None
            Simulation time [s] to display, or ``None`` to hide the label.
        """
        # Remove the existing label first (named replacement handles updates).
        if self._time_label_actor is not None:
            try:
                self._pl.remove_actor(self._time_label_actor)
            except Exception:  # noqa: BLE001
                pass
            self._time_label_actor = None

        if t is None:
            return

        minutes = t / 60.0
        text = f"t = {t:.0f} s" if minutes < 1.0 else f"t = {t:.0f} s  ({minutes:.1f} min)"

        try:
            self._time_label_actor = self._pl.add_text(
                text,
                position="upper_left",
                font_size=14,
                color="white",
                shadow=True,
                name=_TIME_LABEL_NAME,
            )
        except Exception:  # noqa: BLE001 — headless or unsupported plotter
            pass

    def _build_nodal_pv_mesh(self, T_field: Any, t_idx: int) -> "pv.PolyData | None":
        """
        Assemble a PyVista PolyData from the solver surface mesh nodes/quads with
        per-vertex temperatures from T_field.T_section[eid][t_idx].

        Node positions are transformed from global model coords to scene coords by
        subtracting the model centroid (same shift applied by build_model_mesh).
        Quad faces from each element are concatenated with corrected index offsets.

        Returns None if no elements have both nodal_geometry and T_section data.
        """
        ng: dict = getattr(T_field, "nodal_geometry", {})
        all_pts:   list[np.ndarray] = []
        all_faces: list[int]        = []
        all_temps: list[np.ndarray] = []
        offset = 0

        for eid, (nodes_global, quads) in ng.items():
            if eid not in T_field.T_section:
                continue
            pts = np.asarray(nodes_global, dtype=float) - self._centroid
            T_nodes = T_field.T_section[eid][t_idx]          # (n_nodes,)
            all_pts.append(pts)
            all_temps.append(np.asarray(T_nodes, dtype=float))
            for q in quads:
                all_faces.extend([4,
                                   int(q[0]) + offset, int(q[1]) + offset,
                                   int(q[2]) + offset, int(q[3]) + offset])
            offset += len(pts)

        if not all_pts:
            return None

        mesh = pv.PolyData(
            np.vstack(all_pts),
            faces=np.array(all_faces, dtype=np.intp),
        )
        mesh.point_data["temperature_C"] = np.concatenate(all_temps)
        return mesh

    def _remove_nodal_actor(self) -> None:
        if self._nodal_actor is not None:
            try:
                self._pl.remove_actor(self._nodal_actor, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._nodal_actor = None

    def _refresh_actors(self) -> None:
        """Remove then re-add the primary mesh actor with up-to-date settings."""
        # Clear time label whenever temperature colours are no longer active.
        if self._colour_mode != "temperature":
            self._update_time_label(None)

        self._remove_solid_actor()
        self._remove_wire_actor()
        self._remove_nodal_actor()
        # Highlight is element/quad-specific — clear on any mesh rebuild.
        self._remove_highlight()

        if self._render_mode == "wire" and self._wire_mesh is not None:
            self._wire_actor = self._pl.add_mesh(
                self._wire_mesh,
                color=_STEEL_GREY,
                line_width=1.5,
                show_scalar_bar=False,
            )
        elif (
            self._colour_mode == "temperature"
            and self._nodal_mesh is not None
            and self._nodal_mesh.n_points > 0
            and self._T_colour_map.clim is not None
        ):
            # Non-analysed elements: render at full opacity with normal group colours.
            if self._solid_mesh is not None and self._solid_mesh.n_cells > 0:
                analysed_set = set(self._T_field.element_ids) if self._T_field is not None else set()
                eid_arr = self._solid_mesh.cell_data.get("element_id")
                if analysed_set and eid_arr is not None:
                    keep = np.array([eid not in analysed_set for eid in eid_arr])
                    unanalysed_mesh = self._solid_mesh.extract_cells(np.where(keep)[0]) if keep.any() else None
                else:
                    unanalysed_mesh = self._solid_mesh
                if unanalysed_mesh is not None and unanalysed_mesh.n_cells > 0:
                    if self._group_colour_map is not None and self._group_names:
                        bg_kwargs = self._group_colour_map.add_mesh_kwargs()
                    else:
                        bg_kwargs = dict(color=_STRUCTURE_COLOUR, show_scalar_bar=False, show_edges=False)
                    self._solid_actor = self._pl.add_mesh(unanalysed_mesh, **bg_kwargs)
            # Analysed elements: temperature-coloured nodal mesh on top.
            kwargs = self._T_colour_map.add_mesh_kwargs()
            self._nodal_actor = self._pl.add_mesh(
                self._nodal_mesh,
                preference="point",
                **kwargs,
            )
        elif self._render_mode == "section":
            # Inspector mode: replace structural mesh with the FEM mesh so edges
            # and pick-target are the same object — no misalignment.
            if (
                self._inspector_data is not None
                and self._inspector_data.mesh.n_cells > 0
            ):
                self._solid_actor = self._pl.add_mesh(
                    self._inspector_data.mesh,
                    color=_STRUCTURE_COLOUR,
                    show_edges=True,
                    edge_color=_BACKGROUND,
                    line_width=0.8,
                    show_scalar_bar=False,
                )
            elif self._solid_mesh is not None and self._solid_mesh.n_cells > 0:
                self._solid_actor = self._pl.add_mesh(
                    self._solid_mesh,
                    **self._solid_mesh_kwargs(),
                )

        self._pl.render()

    def _remove_solid_actor(self) -> None:
        if self._solid_actor is not None:
            try:
                self._pl.remove_actor(self._solid_actor, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._solid_actor = None

    def _remove_wire_actor(self) -> None:
        if self._wire_actor is not None:
            try:
                self._pl.remove_actor(self._wire_actor, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._wire_actor = None

    def _apply_visibility_filter(self) -> None:
        """
        Rebuild _solid_mesh as the visible subset of _full_solid_mesh.

        Uses element IDs rather than the group_id scalar so that elements
        belonging to multiple groups are handled correctly regardless of which
        group "wins" in _elem_to_group.
        """
        if self._full_solid_mesh is None:
            return

        # Collect element IDs that belong to at least one hidden group
        hidden_eids: set[int] = set()
        if self._model is not None:
            for name, visible in self._group_visible.items():
                if not visible and name in self._model.groups:
                    hidden_eids.update(self._model.groups[name].element_ids)

        if not hidden_eids:
            self._solid_mesh = self._full_solid_mesh
        else:
            eid_arr = self._full_solid_mesh.cell_data["element_id"]
            mask = np.fromiter(
                (int(eid) not in hidden_eids for eid in eid_arr),
                dtype=bool,
                count=len(eid_arr),
            )
            if mask.all():
                self._solid_mesh = self._full_solid_mesh
            else:
                indices = np.where(mask)[0]
                self._solid_mesh = (
                    self._full_solid_mesh.extract_cells(indices)
                    if len(indices) > 0
                    else pv.PolyData()   # all hidden; _refresh_actors will skip
                )

        # Reapply temperature to the rebuilt mesh when a TemperatureField is stored;
        # extract_cells returns a new object so the scalar arrays must be re-written.
        if self._colour_mode == "temperature":
            if self._T_field is not None:
                ng = getattr(self._T_field, "nodal_geometry", {})
                if ng:
                    # Nodal path: rebuild the nodal mesh (solid mesh rebuild above
                    # only affects the structural context overlay).
                    nodal_mesh = self._build_nodal_pv_mesh(self._T_field, self._T_t_idx)
                    if nodal_mesh is not None:
                        self._nodal_mesh = nodal_mesh
                    self._refresh_actors()
                    return
                T_row = self._T_field.T_centroid[self._T_t_idx]
                T_map = {
                    eid: float(T_row[i])
                    for i, eid in enumerate(self._T_field.element_ids)
                }
                self.colour_by_temperature(T_map, clim=self._T_colour_map.clim)
                return  # colour_by_temperature already calls _refresh_actors
            # No stored field (direct colour_by_temperature call) — fall back
            self._colour_mode = "default"
            self._nodal_mesh = None
            self._T_colour_map.clear_clim()

        self._refresh_actors()

    # ── Axis marker ───────────────────────────────────────────────────────────

    def show_axis_marker(self, visible: bool) -> None:
        """Show or hide the 3-D axis origin marker."""
        self._axis_marker_visible = visible
        if visible:
            self._rebuild_axis_marker()
        else:
            self._clear_axis_marker()
        self._pl.render()

    def set_axis_origin_world(self, xyz_world: np.ndarray) -> None:
        """
        Pan the camera so the focal point (and therefore the axis) moves to
        *xyz_world* [global model coords].  Keeps the camera distance/direction.
        """
        self._axis_origin_world = np.asarray(xyz_world, dtype=float).copy()
        scene_pos = self._axis_origin_world - self._centroid
        try:
            old_fp  = np.asarray(self._pl.camera.focal_point, dtype=float)
            old_pos = np.asarray(self._pl.camera.position,    dtype=float)
            delta   = scene_pos - old_fp
            self._pl.camera.focal_point = scene_pos.tolist()
            self._pl.camera.position    = (old_pos + delta).tolist()
        except Exception:  # noqa: BLE001
            pass
        if self._axis_marker_visible:
            self._pl.render()

    def get_axis_origin_world(self) -> np.ndarray:
        """
        Return the current axis origin in global model coordinates [m].

        The axis tracks the camera focal point, so this reads the focal point
        and adds the model centroid offset.
        """
        try:
            fp = np.asarray(self._pl.camera.focal_point, dtype=float)
            return fp + self._centroid
        except Exception:  # noqa: BLE001
            return self._axis_origin_world.copy()

    @property
    def axis_marker_visible(self) -> bool:
        return self._axis_marker_visible

    def set_coord_display_callback(self, callback: Any) -> None:
        """
        Register a callable that receives live coordinate updates on every render.

        Signature: ``callback(world_center: np.ndarray)``

        *world_center* — camera focal point in global model coordinates [m].
        The axis marker (when visible) is always at this position.

        Pass ``None`` to unregister.
        """
        self._coord_display_callback = callback

    def _rebuild_axis_marker(self) -> None:
        """
        Rebuild the KFXView-style cross axis: three long bidirectional lines
        (X=red, Y=green, Z=blue) with small arrowheads at their positive ends
        and a yellow sphere at the origin.

        All geometry is built at the scene origin (0, 0, 0).  Each frame,
        _on_render calls actor.SetPosition(focal_point) to track the camera
        centre so the axis follows panning without being fixed in world space.
        """
        self._clear_axis_marker()

        # Scale relative to model bounding box
        if self._full_solid_mesh is not None and self._full_solid_mesh.n_points > 0:
            b = self._full_solid_mesh.bounds
            span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
            span = max(span, 1.0)
        else:
            span = 10.0

        half_len = span * 10.0   # long lines give "infinite" cross appearance
        tip_at   = span * 0.08   # arrowhead tip distance from origin
        tip_h    = span * 0.022  # cone height
        tip_r    = span * 0.008  # cone base radius
        sph_r    = span * 0.005  # origin sphere radius

        # Save camera position so that adding actors (with their large bounding
        # box from the long lines) doesn't trigger an unwanted auto-zoom.
        try:
            saved_cam = list(self._pl.camera_position)
        except Exception:  # noqa: BLE001
            saved_cam = None

        A = self._axis_marker_actors

        def _add(mesh, **kw):
            try:
                actor = self._pl.add_mesh(mesh, reset_camera=False, **kw)
            except TypeError:
                actor = self._pl.add_mesh(mesh, **kw)
            A.append(actor)

        # ── All geometry at scene origin (0,0,0); SetPosition moves it each frame ──

        # Origin sphere (yellow)
        _add(pv.Sphere(radius=sph_r, center=(0, 0, 0)),
             color="yellow", lighting=False)

        # X axis (red)
        _add(pv.Line(pointa=(-half_len, 0, 0), pointb=(half_len, 0, 0)),
             color=(0.88, 0.15, 0.15), line_width=1.5, lighting=False)
        _add(pv.Cone(center=(tip_at - tip_h * 0.5, 0, 0),
                     direction=(1, 0, 0), height=tip_h, radius=tip_r, resolution=18),
             color=(0.88, 0.15, 0.15), lighting=False)

        # Y axis (green)
        _add(pv.Line(pointa=(0, -half_len, 0), pointb=(0, half_len, 0)),
             color=(0.10, 0.80, 0.10), line_width=1.5, lighting=False)
        _add(pv.Cone(center=(0, tip_at - tip_h * 0.5, 0),
                     direction=(0, 1, 0), height=tip_h, radius=tip_r, resolution=18),
             color=(0.10, 0.80, 0.10), lighting=False)

        # Z axis (blue)
        _add(pv.Line(pointa=(0, 0, -half_len), pointb=(0, 0, half_len)),
             color=(0.15, 0.45, 0.95), line_width=1.5, lighting=False)
        _add(pv.Cone(center=(0, 0, tip_at - tip_h * 0.5),
                     direction=(0, 0, 1), height=tip_h, radius=tip_r, resolution=18),
             color=(0.15, 0.45, 0.95), lighting=False)

        # Restore camera to prevent zoom-out caused by the large actor bbox
        if saved_cam is not None:
            try:
                self._pl.camera_position = saved_cam
            except Exception:  # noqa: BLE001
                pass

        # Position actors at the current camera focal point immediately
        try:
            fp = self._pl.camera.focal_point
            fx, fy, fz = float(fp[0]), float(fp[1]), float(fp[2])
            for actor in A:
                actor.SetPosition(fx, fy, fz)
        except Exception:  # noqa: BLE001
            pass

    def _clear_axis_marker(self) -> None:
        for actor in self._axis_marker_actors:
            try:
                self._pl.remove_actor(actor)
            except Exception:  # noqa: BLE001
                pass
        self._axis_marker_actors.clear()

    # ── Coordinate picking (for fire-zone centre placement) ───────────────────

    def enable_coord_picking(
        self,
        callback: Any,
        *,
        once: bool = True,
    ) -> None:
        """
        Enable world-coordinate point picking on visible mesh surfaces.

        Parameters
        ----------
        callback : callable(np.ndarray)
            Called with the picked point in **global model coordinates** [m].
        once : bool
            If True the callback fires exactly once; picking is then disabled
            and normal cell picking is restored automatically.
        """
        # Temporarily disable cell picking
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass

        self._coord_pick_callback = callback
        self._coord_pick_once = once

        self._pl.enable_point_picking(
            callback=self._on_raw_point_picked,
            show_message=False,
            show_point=True,
            point_size=10,
            color="yellow",
            left_clicking=True,
        )

    def disable_coord_picking(self) -> None:
        """Stop coordinate picking and restore normal element picking."""
        self._coord_pick_callback = None
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass
        if self._pick_callback is not None:
            self.enable_picking(self._pick_callback)

    def _on_raw_point_picked(self, *args: Any) -> None:
        """Internal pyvista callback — converts scene coords to world coords."""
        if not args or args[0] is None:
            return
        point = args[0]
        try:
            pt = np.asarray(point, dtype=float).flatten()[:3]
            if len(pt) < 3:
                return
        except (TypeError, ValueError):
            return

        world_pt = pt + self._centroid
        cb = self._coord_pick_callback

        if self._coord_pick_once:
            self._coord_pick_callback = None
            # Defer picking teardown until after this VTK callback returns
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self.disable_coord_picking)
            except Exception:  # noqa: BLE001
                self.disable_coord_picking()

        if cb is not None:
            cb(world_pt)

    def _on_scene_tick(self) -> None:
        """
        30 Hz timer tick (QTimer).

        Moves all axis actors to the camera focal point so the cross stays
        pinned at the screen centre regardless of panning.  Also drives the
        status-bar coordinate display.
        """
        try:
            fp = self._pl.camera.focal_point
        except Exception:  # noqa: BLE001
            return

        if self._axis_marker_visible and self._axis_marker_actors:
            fx, fy, fz = float(fp[0]), float(fp[1]), float(fp[2])
            for actor in self._axis_marker_actors:
                try:
                    actor.SetPosition(fx, fy, fz)
                except Exception:  # noqa: BLE001
                    pass

        if self._coord_display_callback is not None:
            try:
                world_center = np.asarray(fp, dtype=float) + self._centroid
                self._coord_display_callback(world_center)
            except Exception:  # noqa: BLE001
                pass

    # ── Private (model / actor management) ────────────────────────────────────

    def _clear_model_actors(self) -> None:
        self._remove_solid_actor()
        self._remove_wire_actor()
        self._remove_nodal_actor()
        self._nodal_mesh = None
        self._clear_element_labels()
        self._clear_axis_marker()
        if self._analysis_mesh_actor is not None:
            try:
                self._pl.remove_actor(self._analysis_mesh_actor)
            except Exception:  # noqa: BLE001
                pass
            self._analysis_mesh_actor = None
        self._full_solid_mesh = None
        self._solid_mesh = None
        self._wire_mesh  = None
        self._model      = None
