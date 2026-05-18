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

from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
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
from fahts.gui.panels.model_tree_panel import ModelTreePanel
from fahts.gui.panels.properties_panel import PropertiesPanel
from fahts.gui.panels.results_panel import ResultsPanel
from fahts.renderer.scene_manager import SceneManager

log = logging.getLogger(__name__)


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
        self._post_processor: object = None        # PostProcessor (Task 3.9)

        # Animation state (Task 4.4)
        self._anim_playing: bool = False
        self._anim_t_idx: int = 0
        self._anim_fps: int = 10
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
        ):
            _a.setEnabled(False)
        self._action_threshold_overlay.setChecked(False)
        self._scene.set_threshold_overlay(False)

        self._model = model
        self._scene.load_model(model)
        self._scene.enable_picking(self._on_element_picked)

        # Re-apply colour mode chosen before this load
        if self._action_colour_group.isChecked():
            self._scene.colour_by_group()

        self._action_colour_group.setEnabled(True)
        self._action_mode_wire.setEnabled(True)
        self._action_mode_section.setEnabled(True)
        self._action_reset_cam.setEnabled(True)
        self._action_add_fire_zone.setEnabled(True)
        self._action_axis_marker.setEnabled(True)
        self._action_set_axis.setEnabled(True)
        self._action_screenshot.setEnabled(True)
        # Export BC only enabled once fire zones exist
        self._action_export_bc.setEnabled(bool(self._fire_sources))

        stats = (
            f"Model: {path.name}  |  "
            f"{model.n_nodes} nodes  |  "
            f"{model.n_elements} elements  |  "
            f"{len(model.groups)} groups"
        )
        self._status(stats)
        self.model_loaded.emit(model)
        self._add_to_recent(path)
        log.info("Loaded %s", path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle("FAHTS — Fire Analysis and Heat Transfer Software")
        self.resize(1400, 900)

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
        self._action_cmap_inferno.setChecked(True)
        self._action_cmap_jet = QAction("&Jet", self)
        self._action_cmap_jet.setCheckable(True)
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

        view_m.addSeparator()
        view_m.addSection("Axis")
        view_m.addAction(self._action_axis_marker)
        view_m.addAction(self._action_set_axis)

        # Model
        model_m = mb.addMenu("&Model")
        model_m.addAction(self._action_model_summary)

        # Heat
        heat_m = mb.addMenu("&Heat")
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
        help_m = mb.addMenu("&Help")
        help_m.addAction(self._action_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

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

    def _build_animation_toolbar(self) -> None:
        """
        Build the animation playback toolbar (shown at the bottom when results exist).

        Layout: [⏮][⏪][▶/⏸][⏩][⏭]  FPS: [combo]  [═══slider═══]  "Step N/M  |  t = X s"
        """
        tb = QToolBar("Animation", self)
        tb.setMovable(False)
        tb.setObjectName("anim_toolbar")
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, tb)
        self._anim_toolbar = tb

        # ── Playback buttons ──────────────────────────────────────────────────
        self._action_anim_first = QAction("⏮", self)
        self._action_anim_first.setToolTip("Go to first time step  [Home]")
        self._action_anim_first.setShortcut(QKeySequence(Qt.Key.Key_Home))
        self._action_anim_first.triggered.connect(self._on_anim_first)

        self._action_anim_prev = QAction("⏪", self)
        self._action_anim_prev.setToolTip("Step back one time step  [←]")
        self._action_anim_prev.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self._action_anim_prev.triggered.connect(self._on_anim_prev)

        self._action_anim_play = QAction("▶", self)
        self._action_anim_play.setToolTip("Play / Pause animation  [Space]")
        self._action_anim_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self._action_anim_play.triggered.connect(self._on_anim_play_pause)

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
        tb.addAction(self._action_anim_play)
        tb.addAction(self._action_anim_next)
        tb.addAction(self._action_anim_last)
        tb.addSeparator()

        # ── FPS selector ──────────────────────────────────────────────────────
        tb.addWidget(QLabel(" FPS: "))
        self._anim_speed_combo = QComboBox()
        self._anim_speed_combo.addItems(["1", "2", "5", "10", "15", "20"])
        self._anim_speed_combo.setCurrentText("10")
        self._anim_speed_combo.setFixedWidth(52)
        self._anim_speed_combo.setToolTip("Animation playback speed (frames per second)")
        self._anim_speed_combo.currentTextChanged.connect(self._on_anim_speed_changed)
        tb.addWidget(self._anim_speed_combo)
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

        # ── Timer for playback ────────────────────────────────────────────────
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)

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
        left_splitter.addWidget(self._results_panel)
        left_splitter.setSizes([280, 130, 110, 230])

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

        self.setCentralWidget(main_splitter)

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

    def _update_coord_display(self, world_center: "np.ndarray") -> None:
        """
        Called on every pyvista render to update the status bar coordinate display.

        *world_center* is the camera focal point in global model coordinates [m].
        The axis marker (when visible) always sits at this position.
        """
        import numpy as np  # local to avoid circular import at module level
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

    def _on_toggle_threshold_overlay(self, checked: bool) -> None:
        """Enable or disable the 660 °C critical-temperature threshold overlay."""
        self._scene.set_threshold_overlay(checked)

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
        from fahts.core.heat.bc.view_factor import exposed_element_ids
        from fahts.gui.dialogs.run_analysis_dialog import RunAnalysisDialog

        if self._model is None:
            return

        fire_zones = [s for s in self._fire_sources if isinstance(s, FireZone)]
        rad_balls  = [s for s in self._fire_sources if isinstance(s, RadiationBall)]
        exp_eids: set[int] = set()
        if fire_zones:
            exp_eids |= exposed_element_ids(
                self._model.elements, fire_zones, self._model.nodes
            )
        for ball in rad_balls:
            exp_eids |= ball.exposed_element_ids(
                self._model.elements, self._model.nodes
            ).keys()

        dlg = RunAnalysisDialog(n_exposed=len(exp_eids), parent=self)
        dlg.config_accepted.connect(self._on_analysis_config_accepted)
        dlg.exec()

    def _on_analysis_config_accepted(self, config: object) -> None:
        """Start the analysis worker and show a progress dialog."""
        from PyQt6.QtWidgets import QProgressDialog
        from fahts.core.results.analysis_config import AnalysisConfig
        from fahts.gui.analysis_worker import AnalysisWorker

        cfg: AnalysisConfig = config  # type: ignore[assignment]
        log.info("Starting analysis: %s", cfg.summary())

        self._worker = AnalysisWorker(self._model, self._fire_sources, cfg)

        # Modal progress dialog
        n_est = max(len(cfg.element_ids), 1) if cfg.element_ids else None
        self._progress_dlg = QProgressDialog(
            "Preparing analysis…", "Cancel", 0, n_est or 0, self
        )
        self._progress_dlg.setWindowTitle("Heat Transfer Analysis")
        self._progress_dlg.setMinimumWidth(380)
        self._progress_dlg.setMinimumDuration(0)
        self._progress_dlg.setValue(0)
        self._progress_dlg.canceled.connect(self._worker.cancel)

        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.cancelled.connect(self._on_analysis_cancelled)

        self._worker.start()
        self._status("Analysis running…")

    def _on_analysis_progress(self, current: int, total: int, eid: int) -> None:
        """Update the progress dialog during the run."""
        dlg = self._progress_dlg
        if dlg is None:
            return
        dlg.setMaximum(total)
        dlg.setValue(current)
        if eid >= 0:
            dlg.setLabelText(
                f"Solving element {eid}  ({current + 1} of {total})…"
            )

    def _on_analysis_finished(self, result: object) -> None:
        """Store results, build PostProcessor, and show summary."""
        from fahts.core.results.post_processor import PostProcessor
        from fahts.core.results.temperature_field import TemperatureField
        tf: TemperatureField = result  # type: ignore[assignment]
        self._last_result = tf
        self._post_processor = PostProcessor(tf)

        if self._progress_dlg is not None:
            self._progress_dlg.close()
            self._progress_dlg = None

        summary = self._post_processor.summary_text()
        status_line = (
            f"Analysis complete — {tf.n_elements} element(s), "
            f"{tf.n_steps} steps.  "
            f"Critical (≥ 600 °C): {len(tf.critical_elements())}."
        )
        self._status(status_line)
        log.info("%s\n%s", status_line, summary)

        self._anim_show(tf)
        self._action_export_peak_csv.setEnabled(True)
        self._action_export_history_csv.setEnabled(True)
        self._action_export_vtk.setEnabled(True)
        self._action_export_excel.setEnabled(True)
        self._action_save_animation.setEnabled(True)
        self._action_threshold_overlay.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", summary)

    def _on_analysis_error(self, message: str) -> None:
        """Show an error dialog if the worker raises an exception."""
        if self._progress_dlg is not None:
            self._progress_dlg.close()
            self._progress_dlg = None
        self._status("Analysis failed.")
        log.error("Analysis error: %s", message)
        QMessageBox.critical(self, "Analysis Error", message)

    def _on_analysis_cancelled(self) -> None:
        """Clean up after user cancellation."""
        if self._progress_dlg is not None:
            self._progress_dlg.close()
            self._progress_dlg = None
        self._status("Analysis cancelled.")
        log.info("Analysis cancelled by user.")

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
        """Populate and show the animation toolbar for *tf* (TemperatureField)."""
        n = len(tf.times)
        self._anim_slider.blockSignals(True)
        self._anim_slider.setRange(0, max(n - 1, 0))
        self._anim_slider.setValue(0)
        self._anim_slider.blockSignals(False)
        self._anim_t_idx = 0
        self._anim_playing = False
        self._action_anim_play.setText("▶")
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
        """Pause animation without hiding the toolbar."""
        self._anim_playing = False
        if hasattr(self, "_anim_timer"):
            self._anim_timer.stop()
        if hasattr(self, "_action_anim_play"):
            self._action_anim_play.setText("▶")

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

    def _on_anim_play_pause(self) -> None:
        """Toggle play / pause.  Starts the timer from the current step."""
        if self._anim_playing:
            self._anim_stop()
        else:
            self._anim_playing = True
            self._action_anim_play.setText("⏸")
            interval_ms = max(1, round(1000 / self._anim_fps))
            self._anim_timer.start(interval_ms)

    def _on_anim_tick(self) -> None:
        """QTimer tick — advance one step; stop when the last step is reached."""
        next_idx = self._anim_t_idx + 1
        if next_idx >= self._anim_n_steps():
            self._anim_stop()
            return
        self._anim_go_to(next_idx)

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

    def _on_anim_speed_changed(self, text: str) -> None:
        """Update playback FPS when the combobox selection changes."""
        try:
            self._anim_fps = int(text)
        except ValueError:
            self._anim_fps = 10
        if self._anim_playing:
            self._anim_timer.setInterval(max(1, round(1000 / self._anim_fps)))

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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._anim_stop()
        self._plotter.close()
        super().closeEvent(event)
