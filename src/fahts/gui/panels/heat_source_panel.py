"""
HeatSourcePanel — manages a mixed list of FireZone and RadiationBall objects.

Signals
-------
sources_changed(list)
    Emitted whenever the source list changes.  The list contains a mix of
    FireZone and/or RadiationBall objects.
pick_viewport_requested(object)
    Emitted with the active dialog when the user wants to pick a centre from
    the 3-D viewport.
use_axis_origin_requested(object)
    Emitted with the active dialog when the user wants to snap the centre to
    the current axis marker origin.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.heat.sources.rad_ball import RadiationBall
from fahts.gui.dialogs.fire_zone_dialog import FireZoneDialog
from fahts.gui.dialogs.rad_ball_dialog import RadiationBallDialog

log = logging.getLogger(__name__)

_SOURCE_LABELS = {
    FireZone: "[Zone]",
    RadiationBall: "[Ball]",
}


class HeatSourcePanel(QWidget):
    """Widget that manages a mixed list of FireZone and RadiationBall sources."""

    sources_changed: pyqtSignal = pyqtSignal(list)
    pick_viewport_requested: pyqtSignal = pyqtSignal(object)   # dialog ref
    use_axis_origin_requested: pyqtSignal = pyqtSignal(object)  # dialog ref

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources: list[FireZone | RadiationBall] = []
        self._active_dialog: FireZoneDialog | RadiationBallDialog | None = None
        self._edit_row: int = -1
        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def sources(self) -> list[FireZone | RadiationBall]:
        return list(self._sources)

    @property
    def zones(self) -> list[FireZone]:
        """Backward-compat: return only FireZone objects."""
        return [s for s in self._sources if isinstance(s, FireZone)]

    def set_sources(self, sources: list[FireZone | RadiationBall]) -> None:
        self._sources = list(sources)
        self._refresh_list()
        self.sources_changed.emit(list(self._sources))

    def clear(self) -> None:
        self._sources.clear()
        self._refresh_list()
        self.sources_changed.emit([])

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        box = QGroupBox("Heat Sources")
        v = QVBoxLayout(box)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._on_edit_double_click)
        v.addWidget(self._list)

        # Add row: two buttons for the two source types
        add_row = QHBoxLayout()
        self._btn_add_zone = QPushButton("Add Zone…")
        self._btn_add_ball = QPushButton("Add Ball…")
        self._btn_add_zone.setToolTip("Add a rectangular fire zone")
        self._btn_add_ball.setToolTip("Add a radiation ball (spherical prescribed-flux source)")
        self._btn_add_zone.clicked.connect(self._on_add_zone)
        self._btn_add_ball.clicked.connect(self._on_add_ball)
        add_row.addWidget(self._btn_add_zone)
        add_row.addWidget(self._btn_add_ball)
        v.addLayout(add_row)

        # Edit / Delete row
        edit_row = QHBoxLayout()
        self._btn_edit   = QPushButton("Edit…")
        self._btn_delete = QPushButton("Delete")
        self._btn_edit.setEnabled(False)
        self._btn_delete.setEnabled(False)
        self._btn_edit.clicked.connect(self._on_edit_selected)
        self._btn_delete.clicked.connect(self._on_delete)
        edit_row.addWidget(self._btn_edit)
        edit_row.addWidget(self._btn_delete)
        v.addLayout(edit_row)

        outer.addWidget(box)
        self._list.currentRowChanged.connect(self._on_selection_changed)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_add_zone(self) -> None:
        self._open_zone_dialog(existing=None, row=-1)

    def _on_add_ball(self) -> None:
        self._open_ball_dialog(existing=None, row=-1)

    def _on_edit_selected(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._sources)):
            return
        source = self._sources[row]
        if isinstance(source, FireZone):
            self._open_zone_dialog(existing=source, row=row)
        else:
            self._open_ball_dialog(existing=source, row=row)

    def _on_edit_double_click(self, _: QListWidgetItem) -> None:
        self._on_edit_selected()

    def _on_delete(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._sources)):
            return
        name = self._sources[row].name
        reply = QMessageBox.question(
            self, "Delete Heat Source", f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sources.pop(row)
            self._refresh_list()
            log.info("Deleted heat source '%s'", name)
            self.sources_changed.emit(list(self._sources))

    def _on_selection_changed(self, row: int) -> None:
        has_sel = 0 <= row < len(self._sources)
        self._btn_edit.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel)

    def _on_source_accepted(self, source: FireZone | RadiationBall) -> None:
        """Called when the active non-modal dialog emits its accepted signal."""
        if self._edit_row >= 0:
            self._sources[self._edit_row] = source
            self._edit_row = -1
            log.info("Edited heat source: %s", source)
        else:
            self._sources.append(source)
            self._list.setCurrentRow(len(self._sources) - 1)
            log.info("Added heat source: %s", source)
        self._refresh_list()
        self.sources_changed.emit(list(self._sources))
        self._active_dialog = None

    def _on_dialog_finished(self, _result: int) -> None:
        self._active_dialog = None
        self._edit_row = -1

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_zone_dialog(self, existing: FireZone | None, row: int) -> None:
        if self._active_dialog is not None and self._active_dialog.isVisible():
            self._active_dialog.raise_()
            self._active_dialog.activateWindow()
            return

        dlg = FireZoneDialog(parent=self, existing=existing)
        self._active_dialog = dlg
        self._edit_row = row

        dlg.zone_accepted.connect(self._on_source_accepted)
        dlg.finished.connect(self._on_dialog_finished)
        dlg.pick_viewport_requested.connect(
            lambda: self.pick_viewport_requested.emit(dlg)
        )
        dlg.use_axis_origin_requested.connect(
            lambda: self.use_axis_origin_requested.emit(dlg)
        )
        dlg.show()
        dlg.raise_()

    def _open_ball_dialog(self, existing: RadiationBall | None, row: int) -> None:
        if self._active_dialog is not None and self._active_dialog.isVisible():
            self._active_dialog.raise_()
            self._active_dialog.activateWindow()
            return

        dlg = RadiationBallDialog(parent=self, existing=existing)
        self._active_dialog = dlg
        self._edit_row = row

        dlg.ball_accepted.connect(self._on_source_accepted)
        dlg.finished.connect(self._on_dialog_finished)
        dlg.pick_viewport_requested.connect(
            lambda: self.pick_viewport_requested.emit(dlg)
        )
        dlg.use_axis_origin_requested.connect(
            lambda: self.use_axis_origin_requested.emit(dlg)
        )
        dlg.show()
        dlg.raise_()

    def _refresh_list(self) -> None:
        self._list.clear()
        for source in self._sources:
            prefix = _SOURCE_LABELS.get(type(source), "[?]")
            self._list.addItem(f"{prefix} {source.name}")
        self._on_selection_changed(self._list.currentRow())
