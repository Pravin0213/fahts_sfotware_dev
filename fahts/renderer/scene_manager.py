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
        self._fire_zone_actors: list[Any] = []
        self._rad_ball_actors: list[Any] = []

        # Mode state
        self._render_mode: str = "section"   # 'wire' | 'section'
        self._colour_mode: str = "default"   # 'default' | 'group' | 'temperature'
        self._T_colour_map: TemperatureColourMap = TemperatureColourMap()
        self._T_field: Any = None   # TemperatureField; kept so visibility changes can reapply it
        self._T_t_idx: int = 0      # time-step index last applied

        # Time label actor — text overlay showing t = X s when temperature is active
        self._time_label_actor: Any = None

        # Group colouring + visibility state
        self._group_names:      list[str] = []
        self._group_colour_map: GroupColourMap | None = None
        self._elem_to_group:    dict[int, str] = {}   # eid → group name
        self._group_visible:    dict[str, bool] = {}  # group name → visible flag

        # Picking state
        self._pick_callback: Any = None

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
        # Reset axis origin to model centroid so scene-coord origin = axis position
        self._axis_origin_world = self._centroid.copy()
        if self._axis_marker_visible:
            self._rebuild_axis_marker()

        log.info("Building solid mesh (%d elements)…", model.n_elements)
        self._full_solid_mesh = build_model_mesh(model)
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

    def reset_colour(self) -> None:
        """Return to the default uniform steel-grey colouring."""
        self._colour_mode = "default"
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
        Enable left-click element picking on the solid beam mesh.

        Parameters
        ----------
        callback : callable(element_id: int)
            Called with the integer element ID of the clicked beam face.
            Not called when the click misses all beams.
        """
        self._pick_callback = callback
        # Disable any prior picking session before starting a new one.
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass
        self._pl.enable_cell_picking(
            callback=self._on_cell_picked,
            through=False,
            show=False,
            show_message=False,
        )

    def disable_picking(self) -> None:
        """Disable element picking."""
        self._pick_callback = None
        try:
            self._pl.disable_picking()
        except Exception:  # noqa: BLE001
            pass

    def _on_cell_picked(self, picked: Any) -> None:
        """Internal PyVista cell-picking callback — forward element_id to caller."""
        if self._pick_callback is None or picked is None:
            return
        cell_data = getattr(picked, "cell_data", None)
        if cell_data is None or "element_id" not in cell_data:
            return
        eids = cell_data["element_id"]
        if len(eids) == 0:
            return
        self._pick_callback(int(eids[0]))

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
        Render each radiation ball as a semi-transparent yellow wire sphere.

        Each *ball* must expose:
            .center  — (x, y, z) global centre [m]
            .r1      — inner zone radius [m]
            .r2      — outer zone radius [m]
            .name    — str

        Two concentric spheres are rendered: a solid inner (r1) and a wire outer (r2).
        """
        self.clear_rad_balls()
        for ball in balls:
            cx = float(ball.center[0]) - self._centroid[0]
            cy = float(ball.center[1]) - self._centroid[1]
            cz = float(ball.center[2]) - self._centroid[2]
            centre = (cx, cy, cz)

            # Inner sphere (r1) — solid semi-transparent
            inner = pv.Sphere(radius=float(ball.r1), center=centre)
            a1 = self._pl.add_mesh(
                inner, color=(1.0, 0.85, 0.0), opacity=0.20, style="surface",
            )
            self._rad_ball_actors.append(a1)

            # Outer sphere (r2) — wireframe only
            outer = pv.Sphere(radius=float(ball.r2), center=centre)
            a2 = self._pl.add_mesh(
                outer, color=(1.0, 0.85, 0.0), opacity=0.08,
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

        *T_field* duck-type (matches TemperatureField dataclass from Phase 3):
            .times        np.ndarray  (n_steps,)
            .element_ids  list[int]
            .T_centroid   np.ndarray  (n_steps, n_elems)
        """
        if T_field is None or self._solid_mesh is None:
            return

        times = np.asarray(T_field.times)
        idx = int(np.argmin(np.abs(times - t)))

        self._T_field = T_field
        self._T_t_idx = idx

        T_row = T_field.T_centroid[idx]
        T_map = {eid: float(T_row[i]) for i, eid in enumerate(T_field.element_ids)}

        T_all = np.asarray(T_field.T_centroid)
        lo, hi = float(np.nanmin(T_all)), float(np.nanmax(T_all))
        clim = (lo, hi + 1.0) if hi - lo < 1.0 else (lo, hi)
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

        # default
        return dict(
            color=_STEEL_GREY,
            show_scalar_bar=False,
            show_edges=False,
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

    def _refresh_actors(self) -> None:
        """Remove then re-add the primary mesh actor with up-to-date settings."""
        # Clear time label whenever temperature colours are no longer active.
        if self._colour_mode != "temperature":
            self._update_time_label(None)

        self._remove_solid_actor()
        self._remove_wire_actor()

        if self._render_mode == "wire" and self._wire_mesh is not None:
            self._wire_actor = self._pl.add_mesh(
                self._wire_mesh,
                color=_STEEL_GREY,
                line_width=1.5,
                show_scalar_bar=False,
            )
        elif self._render_mode == "section" and self._solid_mesh is not None:
            if self._solid_mesh.n_cells > 0:   # guard: don't add empty mesh
                self._solid_actor = self._pl.add_mesh(
                    self._solid_mesh,
                    **self._solid_mesh_kwargs(),
                )

        self._pl.render()

    def _remove_solid_actor(self) -> None:
        if self._solid_actor is not None:
            try:
                self._pl.remove_actor(self._solid_actor)
            except Exception:  # noqa: BLE001
                pass
            self._solid_actor = None

    def _remove_wire_actor(self) -> None:
        if self._wire_actor is not None:
            try:
                self._pl.remove_actor(self._wire_actor)
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
                T_row = self._T_field.T_centroid[self._T_t_idx]
                T_map = {
                    eid: float(T_row[i])
                    for i, eid in enumerate(self._T_field.element_ids)
                }
                self.colour_by_temperature(T_map, clim=self._T_colour_map.clim)
                return  # colour_by_temperature already calls _refresh_actors
            # No stored field (direct colour_by_temperature call) — fall back
            self._colour_mode = "default"
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
        self._clear_axis_marker()
        self._full_solid_mesh = None
        self._solid_mesh = None
        self._wire_mesh  = None
        self._model      = None
