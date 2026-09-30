"""Process-vessel workspace: edit a ``VesselCase`` (vessel, contents, fire, relief, ...), run it
and show the results. Lives in its own tab of the main window, next to the 3-D structure view.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                             QVBoxLayout, QWidget)

from fahts.coupling.vessel_case import VesselCase
from fahts.coupling.vessfire_import import case_from_vessfire_deck
from fahts.gui.process.contents_form import ContentsForm
from fahts.gui.process.forms import (AmbientForm, CaseInfoForm, OptionsForm, ReliefForm,
                                     StressRunForm, VesselForm)

log = logging.getLogger(__name__)

CASE_FILTER = "Vessel case (*.vcase.json);;JSON (*.json)"


class ProcessWorkspace(QWidget):
    """Editor for one vessel case. ``current_case()`` returns the case as edited."""

    status_message = pyqtSignal(str)
    title_changed = pyqtSignal(str)

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
        self.run_bar = QHBoxLayout()           # run controls (added by the run step)
        bar.addLayout(self.run_bar)
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
        split.addWidget(self.results_area)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([680, 820])
        root.addWidget(split, 1)

        self.add_form("Case", CaseInfoForm())
        self.add_form("Vessel & material", VesselForm())
        self.add_form("Contents", ContentsForm())
        self.add_form("Relief valves", ReliefForm())
        self.add_form("Surroundings", AmbientForm())
        self.add_form("Stress & run", StressRunForm())
        self.add_form("Model options", OptionsForm())

        contents, vessel = self.form(ContentsForm), self.form(VesselForm)
        contents.btn_check.clicked.connect(
            lambda: contents.check_initial_state(self.current_case()))
        for w in (vessel.D, vessel.L):
            w.valueChanged.connect(lambda *_: contents.set_geometry(vessel.D.value(),
                                                                    vessel.L.value()))
        self.btn_new.clicked.connect(self.new_case)
        self.btn_open.clicked.connect(self._on_open)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as.clicked.connect(self._on_save_as)
        self.btn_import.clicked.connect(self._on_import)
        self.btn_validate.clicked.connect(self.validate)
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

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        name = self._path.name if self._path else "unsaved case"
        self.title_changed.emit(f"{name}{' *' if dirty else ''}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard the changes to this case?")
        return r == QMessageBox.StandardButton.Yes

    def _on_open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open vessel case", "", CASE_FILTER)
        if path:
            try:
                self.open_file(path)
            except (ValueError, KeyError, TypeError, OSError) as e:
                QMessageBox.warning(self, "Open vessel case", f"Could not open {path}:\n{e}")

    def _on_save(self) -> None:
        if self._path is None:
            self._on_save_as()
        else:
            self.save(self._path)

    def _on_save_as(self) -> None:
        name = (self.current_case().name or "case").replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(self, "Save vessel case", f"{name}.vcase.json",
                                              CASE_FILTER)
        if path:
            self.save(path)

    def _on_import(self) -> None:
        if not self._confirm_discard():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "VessFire case folder (Admin.brl, Segment.brl, heatload.scn)")
        if folder:
            try:
                self.import_deck(folder)
            except (OSError, ValueError, KeyError, IndexError) as e:
                QMessageBox.warning(self, "Import VessFire deck", f"Could not import {folder}:\n{e}")
