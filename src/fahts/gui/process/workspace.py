"""Process-vessel workspace: edit a ``VesselCase`` (vessel, contents, fire, relief, ...), run it
and show the results. Lives in its own tab of the main window, next to the 3-D structure view.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMessageBox, QProgressBar, QPushButton, QScrollArea, QSplitter,
                             QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from fahts.coupling.vessel_case import VesselCase
from fahts.coupling.vessfire_import import case_from_vessfire_deck
from fahts.gui.process.contents_form import ContentsForm
from fahts.gui.process.fire_form import FireForm
from fahts.gui.process.results_view import ResultsView
from fahts.gui.process.run_worker import CaseRunWorker
from fahts.gui.process.vessel_view import VesselView
from fahts.gui.process.forms import (AmbientForm, CaseInfoForm, OptionsForm, ReliefForm,
                                     StressRunForm, VesselForm)

log = logging.getLogger(__name__)

CASE_FILTER = "Vessel case (*.vcase.json);;JSON (*.json)"


class ProcessWorkspace(QWidget):
    """Editor for one vessel case. ``current_case()`` returns the case as edited."""

    status_message = pyqtSignal(str)
    title_changed = pyqtSignal(str)
    run_finished = pyqtSignal(object)  # CaseResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._case = VesselCase()
        self._path: Path | None = None
        self._dirty = False
        self._loading = False
        self._forms: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_open = QPushButton("Open…")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save as…")
        self.btn_import = QPushButton("Import VessFire deck…")
        self.btn_validate = QPushButton("Check inputs")
        for b in (self.btn_new, self.btn_open, self.btn_save, self.btn_save_as, self.btn_import,
                  self.btn_validate):
            bar.addWidget(b)
        bar.addStretch(1)
        self.btn_run = QPushButton("▶ Run")
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setFormat("%p %")
        self.progress.setVisible(False)
        for w in (self.progress, self.btn_run, self.btn_stop):
            bar.addWidget(w)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        # section list (left) + the selected section's form
        sections = QHBoxLayout()
        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        self.nav.setSpacing(3)
        self.pages = QStackedWidget()
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        sections.addWidget(self.nav)
        sections.addWidget(self.pages, 1)
        ll.addLayout(sections, 1)
        self.messages = QListWidget()
        self.messages.setWordWrap(True)
        self.messages.setMaximumHeight(120)
        ll.addWidget(QLabel("Input check"))
        ll.addWidget(self.messages)
        split.addWidget(left)
        self.results_area = QWidget()          # results view (added by the run step)
        self.results_layout = QVBoxLayout(self.results_area)
        self.results_placeholder = QLabel("Run the case to see results.")
        self.results_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.results_placeholder)
        self.results = ResultsView()
        self.results.setVisible(False)
        self.results_layout.addWidget(self.results)
        self._worker: CaseRunWorker | None = None
        # right side: 3-D view of the vessel (regions / wall temperature) and the results
        self.right_tabs = QTabWidget()
        self.vessel_view = VesselView()
        self.right_tabs.addTab(self.vessel_view, "3-D view")
        self.right_tabs.addTab(self.results_area, "Results")
        split.addWidget(self.right_tabs)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._refresh_preview)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([680, 820])
        root.addWidget(split, 1)

        self.add_form("Case", CaseInfoForm())
        self.add_form("Vessel & material", VesselForm())
        self.add_form("Contents", ContentsForm())
        self.add_form("Fire", FireForm())
        self.add_form("Relief valves", ReliefForm())
        self.add_form("Surroundings", AmbientForm())
        self.add_form("Stress & run", StressRunForm())
        self.add_form("Model options", OptionsForm())

        contents, vessel = self.form(ContentsForm), self.form(VesselForm)
        contents.btn_check.clicked.connect(
            lambda: contents.check_initial_state(self.current_case()))
        fire = self.form(FireForm)

        def geometry_changed(*_):
            D, L = vessel.D.value(), vessel.L.value()
            contents.set_geometry(D, L)
            fire.set_geometry(D, L, contents.hc.value() + contents.water.value())

        for w in (vessel.D, vessel.L, contents.hc, contents.water):
            w.valueChanged.connect(geometry_changed)
        self.btn_new.clicked.connect(self.new_case)
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save_as.clicked.connect(self.save_as_dialog)
        self.btn_import.clicked.connect(self.import_dialog)
        self.btn_validate.clicked.connect(self.validate)
        self.btn_run.clicked.connect(self.start_run)
        self.btn_stop.clicked.connect(self.stop_run)
        self.load_case(VesselCase())

    # ------------------------------------------------------------------ forms
    def add_form(self, title: str, form, index: int | None = None) -> None:
        """Add an input form (``set_case`` / ``apply`` / ``changed``) as a tab."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(form)
        if index is None:
            index = self.pages.count()
        self.pages.insertWidget(index, area)
        self.nav.insertItem(index, title)
        if self.nav.currentRow() < 0:
            self.nav.setCurrentRow(0)
        self._forms.append(form)
        form.changed.connect(self._on_form_changed)
        if hasattr(self, "_case"):
            self._loading = True
            form.set_case(self._case)
            self._loading = False

    def show_section(self, title: str) -> None:
        for i in range(self.nav.count()):
            if self.nav.item(i).text() == title:
                self.nav.setCurrentRow(i)
                return
        raise KeyError(title)

    def form(self, cls):
        return next(f for f in self._forms if isinstance(f, cls))

    # ------------------------------------------------------------------ case state
    def current_case(self) -> VesselCase:
        """The case with every form's edits applied (a copy)."""
        case = copy.deepcopy(self._case)
        for f in self._forms:
            f.apply(case)
        return case

    def load_case(self, case: VesselCase, path: Path | None = None) -> None:
        self._case = copy.deepcopy(case)
        self._path = Path(path) if path else None
        self._loading = True
        try:
            for f in self._forms:
                f.set_case(self._case)
        finally:
            self._loading = False
        self._set_dirty(False)
        self.validate(quiet=True)
        self._refresh_preview()

    def new_case(self) -> None:
        if self._confirm_discard():
            self.load_case(VesselCase())
            self.status_message.emit("New vessel case")

    def open_file(self, path: Path | str) -> None:
        self.load_case(VesselCase.load(path), path)
        self.status_message.emit(f"Opened {path}")

    def save(self, path: Path | str) -> None:
        case = self.current_case()
        case.save(path)
        self._case, self._path = case, Path(path)
        self._set_dirty(False)
        self.status_message.emit(f"Saved {path}")

    def import_deck(self, folder: Path | str) -> list[str]:
        case, warnings = case_from_vessfire_deck(folder)
        self.load_case(case)
        self._set_dirty(True)
        for w in warnings:
            self._add_message(w, "warning")
        self.status_message.emit(f"Imported VessFire deck {folder}")
        return warnings

    def validate(self, quiet: bool = False) -> list[str]:
        errors = self.current_case().validate()
        self.messages.clear()
        for e in errors:
            self._add_message(e, "error")
        if not errors:
            self._add_message("Inputs OK", "ok")
        if not quiet:
            self.status_message.emit(f"{len(errors)} input error(s)" if errors else "Inputs OK")
        return errors

    # ------------------------------------------------------------------ run
    def start_run(self) -> CaseRunWorker | None:
        """Validate and run the current case in a background thread."""
        if self._worker is not None:
            return None
        if self.validate():
            self.status_message.emit("Fix the input errors before running")
            return None
        case = self.current_case()
        self._worker = CaseRunWorker(case, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)
        self._worker.cancelled.connect(lambda: self._run_done("Run cancelled"))
        self._set_running(True)
        self.status_message.emit(f"Running {case.name} …")
        self._worker.start()
        return self._worker

    def stop_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @property
    def is_running(self) -> bool:
        return self._worker is not None

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.progress.setVisible(running)
        self.progress.setValue(0)
        for b in (self.btn_new, self.btn_open, self.btn_import):
            b.setEnabled(not running)
        self.pages.setEnabled(not running)

    def _on_progress(self, t: float, t_end: float) -> None:
        self.progress.setValue(int(100 * t / max(t_end, 1e-9)))
        self.progress.setFormat(f"{t:.0f} / {t_end:.0f} s")

    def _on_run_finished(self, result) -> None:
        self.results_placeholder.setVisible(False)
        self.results.setVisible(True)
        self.results.show_result(result)
        self.vessel_view.show_result(result)
        if self.right_tabs.currentWidget() is not self.vessel_view:
            self.right_tabs.setCurrentWidget(self.results_area)
        self._run_done(f"Run finished in {result.runtime_s:.1f} s")
        self.run_finished.emit(result)

    def _on_run_error(self, message: str) -> None:
        self._run_done("Run failed")
        self._add_message(f"Run failed: {message}", "error")
        QMessageBox.warning(self, "Run failed", message)

    def _run_done(self, message: str) -> None:
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self._set_running(False)
        self.status_message.emit(message)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def path(self) -> Path | None:
        return self._path

    # ------------------------------------------------------------------ internals
    def _add_message(self, text: str, kind: str) -> None:
        mark = {"error": "✖", "warning": "⚠", "ok": "✔"}[kind]
        self.messages.addItem(QListWidgetItem(f"{mark} {text}"))

    def _on_form_changed(self) -> None:
        if not self._loading:
            self._set_dirty(True)
            self._preview_timer.start()

    def _refresh_preview(self) -> None:
        """Show the edited case's geometry (regions, liquid) in the 3-D view."""
        case = self.current_case()
        v = case.vessel
        if v.inner_diameter_m > 0 and v.length_m > 0:
            try:
                self.vessel_view.show_case(case)
            except (ValueError, ZeroDivisionError) as e:
                log.debug("3-D preview skipped: %s", e)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        name = self._path.name if self._path else "unsaved case"
        self.title_changed.emit(f"{name}{' *' if dirty else ''}")

    def confirm_close(self) -> bool:
        """True if the workspace may be closed (asks about a running run / unsaved edits)."""
        if self.is_running:
            r = QMessageBox.question(self, "Run in progress", "Stop the running simulation?")
            if r != QMessageBox.StandardButton.Yes:
                return False
            self.stop_run()
            if self._worker is not None:
                self._worker.wait()
        return self._confirm_discard()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard the changes to this case?")
        return r == QMessageBox.StandardButton.Yes

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open vessel case", "", CASE_FILTER)
        if path:
            try:
                self.open_file(path)
            except (ValueError, KeyError, TypeError, OSError) as e:
                QMessageBox.warning(self, "Open vessel case", f"Could not open {path}:\n{e}")

    def save_current(self) -> None:
        if self._path is None:
            self.save_as_dialog()
        else:
            self.save(self._path)

    def save_as_dialog(self) -> None:
        name = (self.current_case().name or "case").replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(self, "Save vessel case", f"{name}.vcase.json",
                                              CASE_FILTER)
        if path:
            self.save(path)

    def import_dialog(self) -> None:
        if not self._confirm_discard():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "VessFire case folder (Admin.brl, Segment.brl, heatload.scn)")
        if folder:
            try:
                self.import_deck(folder)
            except (OSError, ValueError, KeyError, IndexError) as e:
                QMessageBox.warning(self, "Import VessFire deck", f"Could not import {folder}:\n{e}")
