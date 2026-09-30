"""
Phase 1.6–1.9 — main_window.py
QMainWindow integrating the PyVista 3-D viewport and FAHTS controls.

Layout
------
  Menu bar  : File | View | Model | Help
  Toolbar   : Open | Reset Camera | [Section | Wire] | [Default | By Group]
  Central   : QSplitter( left_panel | QtInteractor )
  Left panel: ModelTreePanel (Phase 1.7) + PropertiesPanel (Phase 1.8)
  Status bar: model stats after load
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from fahts.core.heat.bc.view_factor import compute_all_bcs, exposed_element_ids
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.core.io.results_writer import (
    export_bc_summary_csv,
    export_bc_summary_multi_time,
    export_peak_temperature_csv,
    export_temperature_history_csv,
    export_results_vtk,
    export_results_excel,
)
from fahts.core.io.usfos_reader import read_usfos_fem
from fahts.core.model.fem_model import FEMModel
from fahts.gui.panels.heat_source_panel import HeatSourcePanel
from fahts.gui.process.workspace import ProcessWorkspace
from fahts.gui.panels.mesh_inspector_panel import MeshInspectorPanel
from fahts.gui.panels.model_tree_panel import ModelTreePanel
from fahts.gui.panels.properties_panel import PropertiesPanel
from fahts.gui.panels.results_panel import ResultsPanel
from fahts.renderer.scene_manager import SceneManager

log = logging.getLogger(__name__)


WINDOW_TITLE = "Vessel Thermal and Rupture Solver"
TAB_HEAT = "Heat Transfer Solver"
TAB_RUPTURE = "Vessel Rupture Solver"


class MainWindow(QMainWindow):
    """
    Top-level application window for FAHTS.

    Signals
    -------
    model_loaded(FEMModel)
        Emitted after a .fem file is successfully read and rendered.
        Consumers (model tree, properties panel) connect here.
    """

    model_loaded: pyqtSignal = pyqtSignal(object)
    element_picked: pyqtSignal = pyqtSignal(int)

    _MAX_RECENT: int = 10

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model: FEMModel | None = None
        self._recent_files: list[Path] = []
        self._fire_sources: list[FireZone | RadiationBall] = []
        self._pending_pick_dialog: object = None   # FireZoneDialog waiting for a coord pick
        self._worker: object = None                # AnalysisWorker (Task 3.8)
        self._progress_dlg: object = None          # QProgressDialog (Task 3.8)
        self._last_result: object = None           # TemperatureField (Task 3.8)
        self._last_config: object = None           # AnalysisConfig — for mesh overlay rebuild
        self._mesh_overlay_eids: list[int] = []   # element IDs currently shown in overlay
        self._post_processor: object = None        # PostProcessor (Task 3.9)
        self._mesh_inspector_data: object = None   # MeshInspectorData (cached)

        # Animation state (Task 4.4)
        self._anim_t_idx: int = 0
        self._anim_fps: int = 10   # used for save-animation only
        self._selected_eid: int | None = None
        self._settings = QSettings("FAHTS", "FAHTS-Solver")
        self._load_recent_files()
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def open_file(self, path: Path | str) -> None:
        """
        Load a USFOS .fem file and render it.

        Called programmatically (e.g. ``python main.py model_file.fem``) and
        also by the File → Open action.
        """
        path = Path(path)
        if not path.exists():
            QMessageBox.critical(self, "File not found", str(path))
            return

        self._status(f"Loading {path.name}…")
        QApplication.processEvents()

        try:
            model = read_usfos_fem(path)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load %s", path)
            QMessageBox.critical(self, "Load error", f"{path.name}:\n{exc}")
            self._status("Load failed.")
            return

        # Clear stale results and hide animation toolbar before loading new model
        self._last_result = None
        self._post_processor = None
        self._selected_eid = None
        self._anim_hide()
        self._results_panel.clear()
        for _a in (
            self._action_export_peak_csv,
            self._action_export_history_csv,
            self._action_export_vtk,
            self._action_export_excel,
            self._action_save_animation,
            self._action_threshold_overlay,
            self._action_legend_range,
        ):
            _a.setEnabled(False)
        self._action_threshold_overlay.setChecked(False)
        self._scene.set_threshold_overlay(False)
        self._scene.reset_legend_clim()
        self._action_show_mesh.setChecked(False)
        self._action_show_mesh.setEnabled(False)
        self._scene.show_analysis_mesh_overlay(None)
        self._last_config = None
        self._mesh_overlay_eids = []
        self._action_inspect_mesh.setChecked(False)
        self._action_inspect_mesh.setEnabled(False)
        self._scene.hide_mesh_inspector()
        self._inspector_panel.clear()
        self._inspector_panel.setVisible(False)
        self._mesh_inspector_data = None

        self._model = model
        self._scene.load_model(model)
        self._scene.enable_picking(self._on_element_picked)

        # Re-apply colour mode chosen before this load
        if self._action_colour_group.isChecked():
            self._scene.colour_by_group()

        self._action_colour_group.setEnabled(True)
        self._action_mode_wire.setEnabled(True)
        self._action_mode_section.setEnabled(True)
        self._action_create_mesh.setEnabled(True)
        self._action_reset_cam.setEnabled(True)
        self._action_add_fire_zone.setEnabled(True)
        self._action_axis_marker.setEnabled(True)
        self._action_set_axis.setEnabled(True)
        self._action_screenshot.setEnabled(True)
        self._action_inspect_mesh.setEnabled(True)
        # Export BC only enabled once fire zones exist
        self._action_export_bc.setEnabled(bool(self._fire_sources))

        stats = (
            f"Model: {path.name}  |  "
            f"{model.n_nodes} nodes  |  "
            f"{model.n_elements} elements  |  "
            f"{len(model.groups)} groups  |  "
            f"Ctrl+click to select element"
        )
        self._status(stats)
        self.model_loaded.emit(model)
        self._add_to_recent(path)
        log.info("Loaded %s", path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)

        self._process_ws = ProcessWorkspace()   # needed by the Process menu and the central tabs
        self._build_actions()
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_animation_toolbar()
        self._build_status_bar()
        # Register live coordinate display — fires on every render via pyvista callback
        self._scene.set_coord_display_callback(self._update_coord_display)

        # Disable model-dependent actions until a file is loaded
        self._action_colour_group.setEnabled(False)
        self._action_mode_wire.setEnabled(False)
        self._action_mode_section.setEnabled(False)
        self._action_reset_cam.setEnabled(False)
        self._action_axis_marker.setEnabled(False)
        self._action_set_axis.setEnabled(False)
        self._action_create_mesh.setEnabled(False)

    def _build_actions(self) -> None:
        # ── File ──────────────────────────────────────────────────────────────
        self._action_open = QAction("&Open…", self)
        self._action_open.setShortcut(QKeySequence.StandardKey.Open)
        self._action_open.setStatusTip("Open a USFOS .fem model file")
        self._action_open.triggered.connect(self._on_open)

        self._action_screenshot = QAction("Save &Screenshot…", self)
        self._action_screenshot.setShortcut("Ctrl+Shift+S")
        self._action_screenshot.setStatusTip("Save a PNG screenshot of the 3-D viewport")
        self._action_screenshot.triggered.connect(self._on_screenshot)
        self._action_screenshot.setEnabled(False)

        self._action_save_animation = QAction("Save &Animation…", self)
        self._action_save_animation.setStatusTip(
            "Render temperature animation to GIF or MP4"
        )
        self._action_save_animation.triggered.connect(self._on_save_animation)
        self._action_save_animation.setEnabled(False)

        self._action_exit = QAction("E&xit", self)
        self._action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self._action_exit.triggered.connect(self.close)

        # ── View — camera ─────────────────────────────────────────────────────
        self._action_reset_cam = QAction("&Reset Camera", self)
        self._action_reset_cam.setShortcut("R")
        self._action_reset_cam.setStatusTip("Fit camera to model bounding box")
        self._action_reset_cam.triggered.connect(lambda: self._scene.reset_camera())

        # ── View — render mode (exclusive) ────────────────────────────────────
        self._action_mode_section = QAction("&Section", self)
        self._action_mode_section.setCheckable(True)
        self._action_mode_section.setChecked(True)
        self._action_mode_section.setStatusTip("Render extruded BOX cross-sections")

        self._action_mode_wire = QAction("&Wire", self)
        self._action_mode_wire.setCheckable(True)
        self._action_mode_wire.setStatusTip("Render beam centre-lines only")

        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addAction(self._action_mode_section)
        self._mode_group.addAction(self._action_mode_wire)
        self._mode_group.triggered.connect(self._on_render_mode_changed)

        # ── View — colour mode (exclusive) ────────────────────────────────────
        self._action_colour_default = QAction("&Default colour", self)
        self._action_colour_default.setCheckable(True)
        self._action_colour_default.setChecked(True)
        self._action_colour_default.setStatusTip("Uniform steel-grey colouring")

        self._action_colour_group = QAction("Colour by &Group", self)
        self._action_colour_group.setCheckable(True)
        self._action_colour_group.setStatusTip("Colour beams by named group")

        self._colour_group_ag = QActionGroup(self)
        self._colour_group_ag.setExclusive(True)
        self._colour_group_ag.addAction(self._action_colour_default)
        self._colour_group_ag.addAction(self._action_colour_group)
        self._colour_group_ag.triggered.connect(self._on_colour_mode_changed)

        # ── Model ─────────────────────────────────────────────────────────────
        self._action_model_summary = QAction("Model &Summary", self)
        self._action_model_summary.setStatusTip("Show model statistics")
        self._action_model_summary.triggered.connect(self._on_model_summary)

        # ── View — temperature colormap (exclusive radio group) ───────────────
        self._action_cmap_inferno = QAction("&Inferno", self)
        self._action_cmap_inferno.setCheckable(True)
        self._action_cmap_jet = QAction("&Jet", self)
        self._action_cmap_jet.setCheckable(True)
        self._action_cmap_jet.setChecked(True)
        self._action_cmap_plasma = QAction("&Plasma", self)
        self._action_cmap_plasma.setCheckable(True)
        self._action_cmap_coolwarm = QAction("&Coolwarm", self)
        self._action_cmap_coolwarm.setCheckable(True)

        self._cmap_action_group = QActionGroup(self)
        self._cmap_action_group.setExclusive(True)
        for _a, _name in (
            (self._action_cmap_inferno, "inferno"),
            (self._action_cmap_jet, "jet"),
            (self._action_cmap_plasma, "plasma"),
            (self._action_cmap_coolwarm, "coolwarm"),
        ):
            _a.setData(_name)
            self._cmap_action_group.addAction(_a)
        self._cmap_action_group.triggered.connect(self._on_temperature_cmap_changed)

        # ── View — threshold overlay (Task 4.9) ───────────────────────────────
        self._action_threshold_overlay = QAction("Critical &Threshold Overlay (660 °C)", self)
        self._action_threshold_overlay.setCheckable(True)
        self._action_threshold_overlay.setChecked(False)
        self._action_threshold_overlay.setStatusTip(
            "Highlight elements at or above 660 °C (EN 1993-1-2 critical temperature) in red"
        )
        self._action_threshold_overlay.toggled.connect(self._on_toggle_threshold_overlay)
        self._action_threshold_overlay.setEnabled(False)

        # ── View — legend fringe range ────────────────────────────────────────
        self._action_legend_range = QAction("Set &Legend Range…", self)
        self._action_legend_range.setStatusTip(
            "Manually set the min/max temperature fringe range for the 3-D legend"
        )
        self._action_legend_range.triggered.connect(self._on_set_legend_range)
        self._action_legend_range.setEnabled(False)

        # ── View — mesh edge overlay ──────────────────────────────────────────
        self._action_show_mesh = QAction("Show &Mesh", self)
        self._action_show_mesh.setCheckable(True)
        self._action_show_mesh.setChecked(False)
        self._action_show_mesh.setStatusTip(
            "Overlay element face edges on the solid section mesh"
        )
        self._action_show_mesh.toggled.connect(self._on_toggle_show_mesh)
        self._action_show_mesh.setEnabled(False)

        # ── View — wall thickness ─────────────────────────────────────────────
        self._action_show_thickness = QAction("Show Wall &Thickness", self)
        self._action_show_thickness.setCheckable(True)
        self._action_show_thickness.setChecked(True)
        self._action_show_thickness.setShortcut("T")
        self._action_show_thickness.setStatusTip(
            "Draw members with their real wall/plate thickness (off = thin USFOS panels)"
        )
        self._action_show_thickness.toggled.connect(self._on_toggle_show_thickness)

        # ── View — mesh inspector ─────────────────────────────────────────────
        self._action_inspect_mesh = QAction("&Inspect Mesh", self)
        self._action_inspect_mesh.setCheckable(True)
        self._action_inspect_mesh.setChecked(False)
        self._action_inspect_mesh.setShortcut("I")
        self._action_inspect_mesh.setStatusTip(
            "Toggle the mesh connectivity inspector — click any quad to see shared nodes"
        )
        self._action_inspect_mesh.toggled.connect(self._on_toggle_inspect_mesh)
        self._action_inspect_mesh.setEnabled(False)

        # ── View — axis marker ────────────────────────────────────────────────
        self._action_axis_marker = QAction("&Axis Marker", self)
        self._action_axis_marker.setCheckable(True)
        self._action_axis_marker.setChecked(False)
        self._action_axis_marker.setStatusTip(
            "Show / hide the 3-D axis origin marker in the viewport"
        )
        self._action_axis_marker.triggered.connect(self._on_toggle_axis_marker)
        self._action_axis_marker.setEnabled(False)

        self._action_set_axis = QAction("&Set Axis Position…", self)
        self._action_set_axis.setStatusTip(
            "Click on the structure to move the axis origin marker"
        )
        self._action_set_axis.triggered.connect(self._on_set_axis_position)
        self._action_set_axis.setEnabled(False)

        # ── Heat ──────────────────────────────────────────────────────────────
        self._action_create_mesh = QAction("&Create Mesh…", self)
        self._action_create_mesh.setShortcut("Ctrl+M")
        self._action_create_mesh.setStatusTip(
            "Set mesh parameters and preview the FEM mesh on the structure"
        )
        self._action_create_mesh.triggered.connect(self._on_create_mesh)
        self._action_create_mesh.setEnabled(False)

        self._action_add_fire_zone = QAction("&Add Fire Zone…", self)
        self._action_add_fire_zone.setStatusTip("Add a rectangular fire zone to the scene")
        self._action_add_fire_zone.triggered.connect(self._on_add_fire_zone)
        self._action_add_fire_zone.setEnabled(False)

        self._action_run_analysis = QAction("&Run Analysis…", self)
        self._action_run_analysis.setShortcut("F5")
        self._action_run_analysis.setStatusTip(
            "Configure and run the heat-transfer analysis for exposed beams"
        )
        self._action_run_analysis.triggered.connect(self._on_run_analysis)
        self._action_run_analysis.setEnabled(False)

        self._action_export_bc = QAction("&Export BC Summary…", self)
        self._action_export_bc.setStatusTip("Export heat-flux BC summary to CSV")
        self._action_export_bc.triggered.connect(self._on_export_bc)
        self._action_export_bc.setEnabled(False)

        # ── Results export (Task 4.7) — enabled after analysis completes ──────
        self._action_export_peak_csv = QAction("Peak Temperature &CSV…", self)
        self._action_export_peak_csv.setStatusTip(
            "Export peak temperature per element to CSV"
        )
        self._action_export_peak_csv.triggered.connect(self._on_export_peak_csv)
        self._action_export_peak_csv.setEnabled(False)

        self._action_export_history_csv = QAction("Temperature &History CSV…", self)
        self._action_export_history_csv.setStatusTip(
            "Export full centroid temperature time-history to CSV"
        )
        self._action_export_history_csv.triggered.connect(self._on_export_history_csv)
        self._action_export_history_csv.setEnabled(False)

        self._action_export_vtk = QAction("&VTK…", self)
        self._action_export_vtk.setStatusTip(
            "Export temperature results as VTK (opens in ParaView)"
        )
        self._action_export_vtk.triggered.connect(self._on_export_vtk)
        self._action_export_vtk.setEnabled(False)

        self._action_export_excel = QAction("&Excel Workbook…", self)
        self._action_export_excel.setStatusTip(
            "Export temperature results to Excel (.xlsx)"
        )
        self._action_export_excel.triggered.connect(self._on_export_excel)
        self._action_export_excel.setEnabled(False)

        # ── Help ──────────────────────────────────────────────────────────────
        self._action_about = QAction("&About FAHTS", self)
        self._action_about.triggered.connect(self._on_about)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_m = mb.addMenu("&File")
        file_m.addAction(self._action_open)

        self._recent_menu = file_m.addMenu("Open &Recent")
        self._rebuild_recent_menu()

        file_m.addSeparator()
        file_m.addAction(self._action_screenshot)
        file_m.addSeparator()
        file_m.addAction(self._action_exit)

        # View
        view_m = mb.addMenu("&View")
        view_m.addAction(self._action_reset_cam)
        view_m.addSeparator()
        view_m.addSection("Render mode")
        view_m.addAction(self._action_mode_section)
        view_m.addAction(self._action_mode_wire)
        view_m.addAction(self._action_show_thickness)
        view_m.addAction(self._action_show_mesh)
        view_m.addAction(self._action_inspect_mesh)
        view_m.addSeparator()
        view_m.addSection("Colour")
        view_m.addAction(self._action_colour_default)
        view_m.addAction(self._action_colour_group)
        view_m.addSeparator()
        cmap_sub = view_m.addMenu("Temperature &Colormap")
        cmap_sub.addAction(self._action_cmap_inferno)
        cmap_sub.addAction(self._action_cmap_jet)
        cmap_sub.addAction(self._action_cmap_plasma)
        cmap_sub.addAction(self._action_cmap_coolwarm)
        view_m.addAction(self._action_threshold_overlay)
        view_m.addAction(self._action_legend_range)

        view_m.addSeparator()
        view_m.addSection("Axis")
        view_m.addAction(self._action_axis_marker)
        view_m.addAction(self._action_set_axis)

        # Model
        model_m = mb.addMenu("&Model")
        model_m.addAction(self._action_model_summary)

        # Heat
        heat_m = mb.addMenu("&Heat")
        heat_m.addAction(self._action_create_mesh)
        heat_m.addSeparator()
        heat_m.addAction(self._action_add_fire_zone)
        heat_m.addSeparator()
        heat_m.addAction(self._action_run_analysis)
        heat_m.addSeparator()
        heat_m.addAction(self._action_export_bc)
        heat_m.addSeparator()
        export_results_m = heat_m.addMenu("Export &Results")
        export_results_m.addAction(self._action_export_peak_csv)
        export_results_m.addAction(self._action_export_history_csv)
        export_results_m.addSeparator()
        export_results_m.addAction(self._action_export_vtk)
        export_results_m.addAction(self._action_export_excel)
        export_results_m.addSeparator()
        export_results_m.addAction(self._action_save_animation)
        self._export_results_menu = export_results_m

        # Help
        # Process vessel (vessel in fire) workspace
        proc_m = mb.addMenu("&Process")
        ws = self._process_ws
        for text, slot, key in (
            ("&New Vessel Case", ws.new_case, "Ctrl+Shift+N"),
            ("&Open Vessel Case…", ws.open_dialog, "Ctrl+Shift+O"),
            ("&Save Vessel Case", ws.save_current, "Ctrl+Shift+S"),
            ("Save Vessel Case &As…", ws.save_as_dialog, None),
            (None, None, None),
            ("&Import VessFire Input Deck…", ws.import_dialog, None),
            (None, None, None),
            ("&Check Inputs", ws.validate, None),
            ("&Run Vessel Case", ws.start_run, "F6"),
            ("S&top Run", ws.stop_run, None),
        ):
            if text is None:
                proc_m.addSeparator()
                continue
            act = QAction(text, self)
            if key:
                act.setShortcut(key)
            act.triggered.connect(lambda _=False, f=slot: (self._show_process_tab(), f()))
            proc_m.addAction(act)

        help_m = mb.addMenu("&Help")
        help_m.addAction(self._action_about)

    def _build_toolbar(self) -> None:
        # lives inside the Heat Transfer Solver tab (placed by _build_central), so the two
        # solver tabs sit directly under the menu bar
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self._main_toolbar = tb

        tb.addAction(self._action_open)
        tb.addSeparator()
        tb.addAction(self._action_reset_cam)
        tb.addSeparator()

        # Render mode label + buttons
        tb.addWidget(QLabel(" Render: "))
        tb.addAction(self._action_mode_section)
        tb.addAction(self._action_mode_wire)
        tb.addSeparator()

        # Colour label + buttons
        tb.addWidget(QLabel(" Colour: "))
        tb.addAction(self._action_colour_default)
        tb.addAction(self._action_colour_group)
        tb.addSeparator()

        # Axis marker
        tb.addWidget(QLabel(" Axis: "))
        tb.addAction(self._action_axis_marker)
        tb.addAction(self._action_set_axis)
        tb.addSeparator()

        # Mesh
        tb.addWidget(QLabel(" Mesh: "))
        tb.addAction(self._action_create_mesh)
        tb.addAction(self._action_show_mesh)
        tb.addAction(self._action_inspect_mesh)

    def _build_animation_toolbar(self) -> None:
        """
        Build the time-navigation toolbar (shown at the bottom when results exist).

        Layout: [⏮][⏪][⏩][⏭]  [═══slider═══]  "Step N/M  |  t = X s"
        Hover over the T-t graph in the sidebar to drive the 3-D view instead of play.
        """
        tb = QToolBar("Animation", self)
        tb.setMovable(False)
        tb.setObjectName("anim_toolbar")
        self._heat_page_layout.addWidget(tb)       # bottom of the Heat Transfer Solver tab
        self._anim_toolbar = tb

        # ── Step navigation buttons ───────────────────────────────────────────
        self._action_anim_first = QAction("⏮", self)
        self._action_anim_first.setToolTip("Go to first time step  [Home]")
        self._action_anim_first.setShortcut(QKeySequence(Qt.Key.Key_Home))
        self._action_anim_first.triggered.connect(self._on_anim_first)

        self._action_anim_prev = QAction("⏪", self)
        self._action_anim_prev.setToolTip("Step back one time step  [←]")
        self._action_anim_prev.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self._action_anim_prev.triggered.connect(self._on_anim_prev)

        self._action_anim_next = QAction("⏩", self)
        self._action_anim_next.setToolTip("Step forward one time step  [→]")
        self._action_anim_next.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self._action_anim_next.triggered.connect(self._on_anim_next)

        self._action_anim_last = QAction("⏭", self)
        self._action_anim_last.setToolTip("Go to last time step  [End]")
        self._action_anim_last.setShortcut(QKeySequence(Qt.Key.Key_End))
        self._action_anim_last.triggered.connect(self._on_anim_last)

        tb.addAction(self._action_anim_first)
        tb.addAction(self._action_anim_prev)
        tb.addAction(self._action_anim_next)
        tb.addAction(self._action_anim_last)
        tb.addSeparator()

        # ── Mesh overlay toggle ───────────────────────────────────────────────
        tb.addAction(self._action_show_mesh)
        tb.addSeparator()

        # ── Time slider (stretches to fill available width) ───────────────────
        self._anim_slider = QSlider(Qt.Orientation.Horizontal)
        self._anim_slider.setRange(0, 0)
        self._anim_slider.setValue(0)
        self._anim_slider.setMinimumWidth(150)
        self._anim_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._anim_slider.setToolTip("Drag to jump to a time step")
        self._anim_slider.valueChanged.connect(self._on_anim_slider_changed)
        tb.addWidget(self._anim_slider)
        tb.addSeparator()

        # ── Time display label ────────────────────────────────────────────────
        self._anim_time_label = QLabel("  t = —")
        self._anim_time_label.setMinimumWidth(200)
        self._anim_time_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        tb.addWidget(self._anim_time_label)

        # Initially hidden until analysis results are available
        tb.setVisible(False)

    def _build_central(self) -> None:
        # 3-D viewport
        self._plotter = QtInteractor(self)
        self._scene = SceneManager(plotter=self._plotter)

        # Left: vertical splitter (tree panel on top, heat sources, properties on bottom)
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self._tree_panel = ModelTreePanel()
        self._tree_panel.set_scene(self._scene)
        self.model_loaded.connect(self._tree_panel.populate)
        left_splitter.addWidget(self._tree_panel)

        self._heat_panel = HeatSourcePanel()
        self._heat_panel.sources_changed.connect(self._on_sources_changed)
        self._heat_panel.pick_viewport_requested.connect(self._on_pick_viewport_requested)
        self._heat_panel.use_axis_origin_requested.connect(self._on_use_axis_origin)
        left_splitter.addWidget(self._heat_panel)

        self._props_panel = PropertiesPanel()
        self._props_placeholder = self._props_panel   # alias kept for tests
        left_splitter.addWidget(self._props_panel)

        self._results_panel = ResultsPanel()
        self._results_panel.time_hovered.connect(self._anim_go_to)
        left_splitter.addWidget(self._results_panel)

        self._inspector_panel = MeshInspectorPanel()
        self._inspector_panel.setVisible(False)
        left_splitter.addWidget(self._inspector_panel)

        left_splitter.setSizes([280, 130, 110, 230, 0])

        # Wrap in a fixed-width frame to match ROADMAP layout
        self._left_panel = QFrame()
        self._left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._left_panel.setMinimumWidth(200)
        self._left_panel.setMaximumWidth(360)
        lp_layout = QVBoxLayout(self._left_panel)
        lp_layout.setContentsMargins(0, 0, 0, 0)
        lp_layout.addWidget(left_splitter)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(self._left_panel)
        main_splitter.addWidget(self._plotter)
        main_splitter.setStretchFactor(0, 0)   # left: fixed
        main_splitter.setStretchFactor(1, 1)   # right: expands
        main_splitter.setSizes([260, 1140])

        # Two solvers, one tab each, directly under the menu bar and across the full width:
        # heat transfer (3-D structure: toolbar, viewport, animation bar) and vessel rupture
        heat_page = QWidget()
        self._heat_page_layout = QVBoxLayout(heat_page)
        self._heat_page_layout.setContentsMargins(0, 0, 0, 0)
        self._heat_page_layout.setSpacing(0)
        self._heat_page_layout.addWidget(self._main_toolbar)
        self._heat_page_layout.addWidget(main_splitter, 1)

        self._workspace_tabs = QTabWidget(self)
        self._workspace_tabs.setObjectName("solverTabs")
        self._workspace_tabs.setDocumentMode(True)
        self._workspace_tabs.tabBar().setExpanding(True)
        self._workspace_tabs.setStyleSheet(
            "#solverTabs > QTabBar::tab { height: 32px; font-weight: bold; color: palette(mid); }"
            "#solverTabs > QTabBar::tab:selected { color: palette(bright-text);"
            " border-bottom: 3px solid palette(highlight); }")
        self._workspace_tabs.addTab(heat_page, TAB_HEAT)
        self._process_ws.status_message.connect(self._status)
        self._process_ws.title_changed.connect(
            lambda t: self._workspace_tabs.setTabText(1, f"{TAB_RUPTURE} — {t}"))
        self._workspace_tabs.addTab(self._process_ws, TAB_RUPTURE)
        self.setCentralWidget(self._workspace_tabs)

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        # Left: general messages (model stats, pick prompts, etc.)
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, stretch=1)

        # Right: live axis + view-centre coordinates (monospace, always visible)
        self._coord_label = QLabel("")
        self._coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding-right: 6px;"
        )
        sb.addPermanentWidget(self._coord_label)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open USFOS model",
            str(Path.home()),
            "USFOS files (*.fem);;All files (*)",
        )
        if path:
            self.open_file(Path(path))

    def _on_clear_recent(self) -> None:
        self._recent_files.clear()
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _on_render_mode_changed(self, action: QAction) -> None:
        mode = "section" if action is self._action_mode_section else "wire"
        self._scene.set_render_mode(mode)

    def _on_colour_mode_changed(self, action: QAction) -> None:
        if action is self._action_colour_group:
            self._scene.colour_by_group()
        else:
            self._scene.reset_colour()

    def _on_model_summary(self) -> None:
        if self._model is None:
            QMessageBox.information(self, "Model Summary", "No model loaded.")
            return
        QMessageBox.information(self, "Model Summary", self._model.summary())

    def _on_element_picked(self, eid: int) -> None:
        """Forwarded from SceneManager picking callback — show element properties."""
        if self._model is None:
            return
        self._selected_eid = eid
        self._props_panel.show_element(eid, self._model)
        self._scene.highlight_element(eid)
        T_section = getattr(self._last_result, "T_section", {}) if self._last_result else {}
        if self._last_result is not None and eid in T_section:
            self._results_panel.show_section(
                eid, self._model, self._last_result, self._anim_t_idx
            )
        else:
            self._results_panel.clear()
        self.element_picked.emit(eid)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About FAHTS",
            "<b>FAHTS</b> — Fire Analysis and Heat Transfer Software<br><br>"
            "Version 0.2 (Phase 2 — Heat Sources)<br>"
            "Reads USFOS .fem structural models and renders them in 3-D.<br>"
            "Fire zones placed in the scene highlight exposed beams and<br>"
            "compute EN 1993-1-2 net heat fluxes.<br><br>"
            "Heat transfer analysis (solver) available in Phase 3.",
        )

    def _update_coord_display(self, world_center) -> None:
        """
        Called on every pyvista render to update the status bar coordinate display.

        *world_center* is the camera focal point in global model coordinates [m].
        The axis marker (when visible) always sits at this position.
        """
        vx, vy, vz = float(world_center[0]), float(world_center[1]), float(world_center[2])
        if self._scene.axis_marker_visible:
            text = f"Axis:  ({vx:10.3f},  {vy:10.3f},  {vz:10.3f}) m"
        else:
            text = f"View:  ({vx:10.3f},  {vy:10.3f},  {vz:10.3f}) m"
        self._coord_label.setText(text)

    def _on_temperature_cmap_changed(self, action: QAction) -> None:
        """Switch the temperature colourmap when the user picks from the View menu."""
        cmap_name: str = action.data()
        self._scene.set_temperature_cmap(cmap_name)

    def _on_create_mesh(self) -> None:
        """Open mesh preview dialog; on accept build overlay for all model elements."""
        from fahts.gui.dialogs.mesh_preview_dialog import MeshPreviewDialog
        dlg = MeshPreviewDialog(parent=self, current_config=self._last_config)
        dlg.mesh_accepted.connect(self._on_mesh_preview_accepted)
        dlg.exec()

    def _on_mesh_preview_accepted(self, config: object) -> None:
        """Store the mesh config, resolve element IDs, and show the overlay."""
        if self._model is None:
            return
        self._last_config = config
        self._mesh_inspector_data = None  # stale — rebuild on next inspector toggle
        # Sync pipe visual geometry to match the chosen circumferential mesh density
        self._scene.set_pipe_sides(config.c_circ)
        # Use all beam elements (any supported section type); filter happens inside builder
        self._mesh_overlay_eids = list(self._model.elements.keys())
        self._action_show_mesh.setEnabled(True)
        # Always show when user explicitly creates the mesh
        self._action_show_mesh.blockSignals(True)
        self._action_show_mesh.setChecked(True)
        self._action_show_mesh.blockSignals(False)
        self._rebuild_analysis_mesh_overlay()
        n_beams = len(self._model.elements)
        n_shells = len(self._model.shell_elements)
        total = n_beams + n_shells
        self._status(f"Mesh preview applied — {total} elements ({n_shells} shells)."
                     if n_shells else f"Mesh preview applied — {total} elements.")

    def _on_toggle_show_thickness(self, checked: bool) -> None:
        """Switch between real-thickness solid rendering and thin mid-surface panels."""
        self._scene.set_show_thickness(checked)

    def _on_toggle_show_mesh(self, checked: bool) -> None:
        """Show or hide the FEM analysis mesh wireframe overlay."""
        if checked:
            self._rebuild_analysis_mesh_overlay()
        else:
            self._scene.show_analysis_mesh_overlay(None)

    def _rebuild_analysis_mesh_overlay(self) -> None:
        """Build the analysis mesh PolyData and pass it to the scene manager."""
        if self._model is None or self._last_config is None or not self._mesh_overlay_eids:
            return
        from fahts.renderer.beam_geometry import build_analysis_mesh_overlay
        overlay = build_analysis_mesh_overlay(
            self._model,
            self._mesh_overlay_eids,
            self._last_config,
            self._model.centroid(),
        )
        self._scene.show_analysis_mesh_overlay(overlay)

    def _on_toggle_inspect_mesh(self, checked: bool) -> None:
        """Show or hide the mesh connectivity inspector overlay."""
        if checked:
            self._ensure_inspector_data()
            if self._mesh_inspector_data is not None:
                self._scene.show_mesh_inspector(
                    self._mesh_inspector_data,
                    self._on_quad_inspected,
                )
                self._inspector_panel.setVisible(True)
        else:
            self._scene.hide_mesh_inspector()
            self._inspector_panel.clear()
            self._inspector_panel.setVisible(False)

    def _ensure_inspector_data(self) -> None:
        """Build inspector data if not already cached; uses last config or defaults."""
        if self._mesh_inspector_data is not None or self._model is None:
            return
        from fahts.renderer.beam_geometry import build_mesh_inspector_data
        self._mesh_inspector_data = build_mesh_inspector_data(
            self._model,
            self._model.centroid(),
            config=self._last_config,
        )

    def _on_quad_inspected(self, beam_eid: int, quad_idx: int) -> None:
        """Populate the inspector panel and highlight the quad when it is clicked."""
        if self._mesh_inspector_data is None:
            return
        data = self._mesh_inspector_data
        quads = data.beam_quads.get(beam_eid)
        if quads is None or quad_idx >= len(quads):
            return
        quad_nodes = quads[quad_idx].tolist()
        gdof_arr = data.beam_node_gdof.get(beam_eid)
        quad_gdofs = [int(gdof_arr[n]) for n in quad_nodes] if gdof_arr is not None else None
        self._inspector_panel.show_quad_info(beam_eid, quad_idx, quad_nodes, quads,
                                             quad_gdofs=quad_gdofs)
        self._scene.highlight_inspector_quad(beam_eid, quad_idx)
        # Also update properties panel so the owning element is identified
        if self._model is not None and beam_eid in self._model.elements:
            self._props_panel.show_element(beam_eid, self._model)

    def _on_toggle_threshold_overlay(self, checked: bool) -> None:
        """Enable or disable the 660 °C critical-temperature threshold overlay."""
        self._scene.set_threshold_overlay(checked)

    def _on_set_legend_range(self) -> None:
        """Open the legend fringe-range dialog and apply the user's choice."""
        from fahts.gui.dialogs.legend_range_dialog import LegendRangeDialog
        current = self._scene.legend_clim
        if current is not None:
            lo, hi = current
        elif self._last_result is not None:
            import numpy as np
            T_all = np.asarray(self._last_result.T_centroid)
            lo = float(np.nanmin(T_all))
            hi = float(np.nanmax(T_all))
        else:
            lo, hi = 20.0, 1000.0
        dlg = LegendRangeDialog(self, current_lo=lo, current_hi=hi)
        dlg.range_accepted.connect(self._scene.set_legend_clim)
        dlg.range_reset.connect(self._scene.reset_legend_clim)
        dlg.exec()

    def _on_toggle_axis_marker(self) -> None:
        """Show or hide the 3-D axis origin marker in the viewport."""
        visible = self._action_axis_marker.isChecked()
        self._scene.show_axis_marker(visible)
        self._action_set_axis.setEnabled(visible and self._model is not None)
        if not visible:
            # Clear the axis part of the status bar immediately
            self._coord_label.setText("")

    def _on_set_axis_position(self) -> None:
        """Enable one-shot point picking to reposition the axis origin marker."""
        if self._model is None:
            return
        self._status("Click on the structure to place the axis origin marker…")
        self._scene.enable_coord_picking(self._on_axis_point_picked, once=True)

    def _on_axis_point_picked(self, xyz_world) -> None:
        """Callback: axis origin pick completed — move the marker."""
        import numpy as np
        self._scene.set_axis_origin_world(np.asarray(xyz_world))
        x, y, z = float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])
        self._status(f"Axis origin set to ({x:.3f}, {y:.3f}, {z:.3f}) m")

    def _on_pick_viewport_requested(self, dialog: object) -> None:
        """
        The fire zone dialog requested a viewport pick for the centre position.

        Enables one-shot coordinate picking; the result is forwarded back to
        the dialog via set_centre().
        """
        if self._model is None:
            return
        self._pending_pick_dialog = dialog
        self._status("Click on the structure to pick the fire zone centre position…")
        self._scene.enable_coord_picking(self._on_firezone_point_picked, once=True)

    def _on_firezone_point_picked(self, xyz_world) -> None:
        """Callback: fire-zone centre pick completed — update the open dialog."""
        import numpy as np
        dlg = self._pending_pick_dialog
        self._pending_pick_dialog = None
        x, y, z = float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])
        self._status(f"Centre picked at ({x:.3f}, {y:.3f}, {z:.3f}) m")
        if dlg is not None and hasattr(dlg, "set_centre"):
            dlg.set_centre(np.asarray(xyz_world))
            dlg.set_pick_mode_active(False)

    def _on_use_axis_origin(self, dialog: object) -> None:
        """
        Copy the current axis marker world position to the fire zone dialog's
        centre fields.  If the axis marker is not visible, inform the user.
        """
        if not self._scene.axis_marker_visible:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Axis Marker Not Visible",
                "Enable the axis marker first:\n  View → Axis Marker  (or toolbar).\n\n"
                "Then position it with  View → Set Axis Position…",
            )
            return
        xyz = self._scene.get_axis_origin_world()
        if hasattr(dialog, "set_centre"):
            dialog.set_centre(xyz)

    def _on_add_fire_zone(self) -> None:
        """Open the fire zone dialog via the heat source panel."""
        self._heat_panel._on_add_zone()  # noqa: SLF001

    def _on_sources_changed(self, sources: list) -> None:
        """React to the heat source panel source list changing."""
        self._fire_sources = sources
        fire_zones = [s for s in sources if isinstance(s, FireZone)]
        rad_balls  = [s for s in sources if isinstance(s, RadiationBall)]

        # Update 3-D scene
        self._scene.show_fire_zones(fire_zones)
        self._scene.show_rad_balls(rad_balls)

        if self._model is not None and sources:
            # Compute exposed beams from all source types
            exp_eids: set[int] = set()
            if fire_zones:
                exp_eids |= exposed_element_ids(
                    self._model.elements, fire_zones, self._model.nodes
                )
            for ball in rad_balls:
                exp_eids |= ball.exposed_element_ids(
                    self._model.elements, self._model.nodes
                ).keys()

            self._scene.highlight_exposed_beams(exp_eids)
            n_zones = len(fire_zones)
            n_balls = len(rad_balls)
            self._status(
                f"Sources: {n_zones} zone(s), {n_balls} ball(s)  |  "
                f"Exposed beams: {len(exp_eids)}"
            )
            log.info(
                "Heat sources updated — %d zone(s), %d ball(s), %d exposed beams",
                n_zones, n_balls, len(exp_eids),
            )
        else:
            self._scene.highlight_exposed_beams(set())
            self._action_export_bc.setEnabled(False)
            if self._model is not None:
                self._status("Heat sources cleared.")

        has_sources = bool(sources) and self._model is not None
        self._action_export_bc.setEnabled(bool(fire_zones) and self._model is not None)
        self._action_run_analysis.setEnabled(has_sources)

    def _on_run_analysis(self) -> None:
        """Open the Run Analysis dialog and emit the config when accepted."""
        from fahts.core.heat.solver.analysis_runner import exposed_analysis_element_ids
        from fahts.gui.dialogs.run_analysis_dialog import RunAnalysisDialog

        if self._model is None:
            return

        # Same screening as run_analysis: beams AND shells, all active source types
        exp_eids = exposed_analysis_element_ids(self._model, self._fire_sources)

        dlg = RunAnalysisDialog(n_exposed=len(exp_eids), parent=self)
        dlg.config_accepted.connect(self._on_analysis_config_accepted)
        dlg.exec()

    def _on_analysis_config_accepted(self, config: object) -> None:
        """Start the analysis worker and show the USFOS-style console dialog."""
        from fahts.core.results.analysis_config import AnalysisConfig
        from fahts.gui.analysis_worker import AnalysisWorker
        from fahts.gui.dialogs.analysis_console_dialog import AnalysisConsoleDialog

        cfg: AnalysisConfig = config  # type: ignore[assignment]
        log.info("Starting analysis: %s", cfg.summary())
        self._last_config = cfg

        self._worker = AnalysisWorker(self._model, self._fire_sources, cfg)

        self._progress_dlg = AnalysisConsoleDialog(self)
        self._progress_dlg.rejected.connect(self._worker.cancel)

        self._worker.log_line.connect(self._progress_dlg.append_line)
        self._worker.progress.connect(self._progress_dlg.set_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.cancelled.connect(self._on_analysis_cancelled)

        self._worker.start()
        self._progress_dlg.show()
        self._status("Analysis running…")

    def _on_analysis_finished(self, result: object) -> None:
        """Store results, update scene, and mark console complete."""
        from fahts.core.results.post_processor import PostProcessor
        from fahts.core.results.temperature_field import TemperatureField
        tf: TemperatureField = result  # type: ignore[assignment]
        self._last_result = tf
        self._post_processor = PostProcessor(tf)

        status_line = (
            f"Analysis complete — {tf.n_elements} element(s), "
            f"{tf.n_steps} steps.  "
            f"Critical (≥ 600 °C): {len(tf.critical_elements())}."
        )
        self._status(status_line)
        log.info(status_line)

        self._anim_show(tf)
        self._results_panel.show_results(tf)
        self._action_export_peak_csv.setEnabled(True)
        self._action_export_history_csv.setEnabled(True)
        self._action_export_vtk.setEnabled(True)
        self._action_export_excel.setEnabled(True)
        self._action_save_animation.setEnabled(True)
        self._action_threshold_overlay.setEnabled(True)
        self._action_legend_range.setEnabled(True)
        self._action_show_mesh.setEnabled(True)
        self._mesh_overlay_eids = list(tf.element_ids)
        if self._action_show_mesh.isChecked():
            self._rebuild_analysis_mesh_overlay()

        if self._progress_dlg is not None:
            self._progress_dlg.set_complete()

    def _on_analysis_error(self, message: str) -> None:
        """Show error state in console dialog."""
        self._status("Analysis failed.")
        log.error("Analysis error: %s", message)
        if self._progress_dlg is not None:
            self._progress_dlg.set_failed(message)
        else:
            QMessageBox.critical(self, "Analysis Error", message)

    def _on_analysis_cancelled(self) -> None:
        """Acknowledge cancellation in console dialog."""
        self._status("Analysis cancelled.")
        log.info("Analysis cancelled by user.")
        if self._progress_dlg is not None:
            self._progress_dlg.set_cancelled()

    def _on_export_bc(self) -> None:
        """Export EN 1993-1-2 BC summary for multiple time steps to CSV."""
        fire_zones = [s for s in self._fire_sources if isinstance(s, FireZone)]
        if self._model is None or not fire_zones:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export BC Summary",
            "bc_summary.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        times = [0.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0]
        bcs_per_time: dict[float, list] = {}
        for t in times:
            bcs = compute_all_bcs(
                self._model.elements,
                fire_zones,
                self._model.nodes,
                t=t,
            )
            if bcs:
                bcs_per_time[t] = bcs

        export_bc_summary_multi_time(bcs_per_time, Path(path))
        self._status(f"BC summary exported → {Path(path).name}")
        QMessageBox.information(
            self, "Export Complete",
            f"BC summary written to:\n{path}\n\n"
            f"Time steps: {sorted(bcs_per_time)} s\n"
            f"Exposed elements per step: "
            f"{[len(v) for v in bcs_per_time.values()]}",
        )

    # ── Results export handlers (Task 4.7) ───────────────────────────────────

    def _on_export_peak_csv(self) -> None:
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Peak Temperature CSV",
            "peak_temperature.csv", "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            export_peak_temperature_csv(self._last_result, Path(path))
            self._status(f"Peak temperature CSV exported → {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_history_csv(self) -> None:
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Temperature History CSV",
            "temperature_history.csv", "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            export_temperature_history_csv(self._last_result, Path(path))
            self._status(f"Temperature history CSV exported → {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_vtk(self) -> None:
        if self._last_result is None or self._model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export VTK",
            "temperature_results.vtk",
            "VTK files (*.vtk);;VTK XML (*.vtp);;All files (*)",
        )
        if not path:
            return
        try:
            export_results_vtk(self._last_result, self._model, Path(path))
            self._status(f"VTK exported → {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_excel(self) -> None:
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel Workbook",
            "temperature_results.xlsx",
            "Excel files (*.xlsx);;All files (*)",
        )
        if not path:
            return
        try:
            export_results_excel(self._last_result, Path(path))
            self._status(f"Excel workbook exported → {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Screenshot / animation export handlers (Task 4.8) ────────────────────

    def _on_screenshot(self) -> None:
        """Save a PNG screenshot of the 3-D viewport."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot",
            "screenshot.png",
            "PNG images (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            self._scene.screenshot(Path(path))
            self._status(f"Screenshot saved → {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Screenshot failed")
            QMessageBox.critical(self, "Screenshot Error", str(exc))

    def _on_save_animation(self) -> None:
        """Render full temperature animation to GIF or MP4."""
        if self._last_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation",
            "animation.gif",
            "GIF animation (*.gif);;MP4 video (*.mp4);;All files (*)",
        )
        if not path:
            return

        from PyQt6.QtWidgets import QProgressDialog  # noqa: PLC0415

        n_steps = len(self._last_result.times)
        progress = QProgressDialog(
            "Rendering animation…", "Cancel", 0, n_steps, self
        )
        progress.setWindowTitle("Save Animation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)
        progress.show()

        cancelled = False

        def _on_frame(cur: int, total: int) -> None:
            nonlocal cancelled
            progress.setValue(cur)
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True
                raise InterruptedError("cancelled")

        try:
            n = self._scene.save_animation(
                self._last_result, path,
                fps=self._anim_fps,
                progress_callback=_on_frame,
            )
            progress.setValue(n_steps)
            progress.close()
            self._status(f"Animation saved → {Path(path).name}")
            QMessageBox.information(
                self, "Animation Saved",
                f"Animation written to:\n{path}\n\n{n} frames @ {self._anim_fps} fps",
            )
        except InterruptedError:
            progress.close()
            self._status("Animation export cancelled.")
        except ImportError as exc:
            progress.close()
            QMessageBox.critical(self, "Missing Dependency", str(exc))
        except Exception as exc:  # noqa: BLE001
            progress.close()
            log.exception("Animation export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Animation helpers (Task 4.4) ──────────────────────────────────────────

    def _anim_n_steps(self) -> int:
        """Return the number of time steps in the current result, or 0."""
        return len(self._last_result.times) if self._last_result is not None else 0

    def _anim_show(self, tf: object) -> None:
        """Populate and show the navigation toolbar for *tf* (TemperatureField)."""
        n = len(tf.times)
        self._anim_slider.blockSignals(True)
        self._anim_slider.setRange(0, max(n - 1, 0))
        self._anim_slider.setValue(0)
        self._anim_slider.blockSignals(False)
        self._anim_t_idx = 0
        self._anim_toolbar.setVisible(True)
        self._anim_go_to(0)

    def _anim_hide(self) -> None:
        """Stop animation and hide the toolbar."""
        self._anim_stop()
        if hasattr(self, "_anim_toolbar"):
            self._anim_toolbar.setVisible(False)
        if hasattr(self, "_anim_time_label"):
            self._anim_time_label.setText("  t = —")

    def _anim_stop(self) -> None:
        """No-op — auto-play removed; navigation is driven by graph hover or step buttons."""

    def _anim_go_to(self, idx: int) -> None:
        """Jump to time step *idx*: update slider, scene, and time label."""
        if self._last_result is None:
            return
        n = self._anim_n_steps()
        idx = max(0, min(idx, n - 1))
        self._anim_t_idx = idx

        self._anim_slider.blockSignals(True)
        self._anim_slider.setValue(idx)
        self._anim_slider.blockSignals(False)

        t = float(self._last_result.times[idx])
        self._scene.update_temperature(t, self._last_result)
        self._results_panel.update_time(idx)

        minutes = t / 60.0
        if minutes < 1.0:
            time_str = f"t = {t:.0f} s"
        else:
            time_str = f"t = {t:.0f} s  ({minutes:.1f} min)"
        self._anim_time_label.setText(f"  Step {idx + 1}/{n}  |  {time_str}")

    def _on_anim_first(self) -> None:
        self._anim_stop()
        self._anim_go_to(0)

    def _on_anim_prev(self) -> None:
        self._anim_stop()
        self._anim_go_to(self._anim_t_idx - 1)

    def _on_anim_next(self) -> None:
        self._anim_stop()
        self._anim_go_to(self._anim_t_idx + 1)

    def _on_anim_last(self) -> None:
        self._anim_stop()
        self._anim_go_to(self._anim_n_steps() - 1)

    def _on_anim_slider_changed(self, value: int) -> None:
        """User dragged the slider — pause and jump to that step."""
        self._anim_stop()
        self._anim_go_to(value)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _status(self, msg: str) -> None:
        self._status_label.setText(msg)

    # ── Recent files ──────────────────────────────────────────────────────────

    def _load_recent_files(self) -> None:
        """Read the persisted recent-files list from QSettings."""
        raw = self._settings.value("recentFiles", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        elif raw is None:
            raw = []
        self._recent_files = [Path(p) for p in raw if p]

    def _save_recent_files(self) -> None:
        """Persist the current recent-files list to QSettings."""
        self._settings.setValue("recentFiles", [str(p) for p in self._recent_files])

    def _add_to_recent(self, path: Path) -> None:
        """Prepend *path* to the recent list, dedup, trim to _MAX_RECENT, save."""
        resolved = path.resolve()
        self._recent_files = [p for p in self._recent_files if p.resolve() != resolved]
        self._recent_files.insert(0, resolved)
        self._recent_files = self._recent_files[: self._MAX_RECENT]
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        """Repopulate the Open Recent submenu from self._recent_files."""
        self._recent_menu.clear()

        if not self._recent_files:
            placeholder = QAction("No recent files", self)
            placeholder.setEnabled(False)
            self._recent_menu.addAction(placeholder)
            return

        for i, path in enumerate(self._recent_files, 1):
            mnemonic = f"&{i}. " if i <= 9 else f"{i}. "
            action = QAction(f"{mnemonic}{path.name}", self)
            action.setStatusTip(str(path))
            if not path.exists():
                action.setEnabled(False)
            action.triggered.connect(
                lambda checked=False, p=path: self.open_file(p)
            )
            self._recent_menu.addAction(action)

        self._recent_menu.addSeparator()
        clear_action = QAction("&Clear Recent", self)
        clear_action.triggered.connect(self._on_clear_recent)
        self._recent_menu.addAction(clear_action)

    def _show_process_tab(self) -> None:
        self._workspace_tabs.setCurrentWidget(self._process_ws)

    def open_process_case(self, path: Path | str) -> None:
        """Open a vessel case file (*.vcase.json) or import a VessFire input-deck folder."""
        path = Path(path)
        self._show_process_tab()
        try:
            if path.is_dir():
                self._process_ws.import_deck(path)
            else:
                self._process_ws.open_file(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.critical(self, "Open vessel case", f"{path}:\n{exc}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._process_ws.confirm_close():
            event.ignore()
            return
        self._anim_stop()
        self._plotter.close()
        super().closeEvent(event)
