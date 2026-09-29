"""
RadiationBallDialog
Qt dialog for placing / editing a RadiationBall in 3-D space.

The dialog is non-modal by design — callers should use show() so the user
can freely move it while interacting with the 3-D viewport.

Signals
-------
ball_accepted(object)          : emitted on OK with the built RadiationBall
pick_viewport_requested()      : user wants to click the viewport to pick centre
use_axis_origin_requested()    : user wants to use the current axis marker position
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fahts.core.heat.sources.rad_ball import RadiationBall


class RadiationBallDialog(QDialog):
    """
    Non-modal dialog for creating or editing a RadiationBall.

    Fields: name, centre X/Y/Z, radius [m], flux [W/m²].
    """

    ball_accepted: pyqtSignal = pyqtSignal(object)       # emits RadiationBall
    pick_viewport_requested: pyqtSignal = pyqtSignal()
    use_axis_origin_requested: pyqtSignal = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        existing: RadiationBall | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Add Radiation Ball" if existing is None else "Edit Radiation Ball"
        )
        self.setMinimumWidth(400)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._build_ui()
        if existing is not None:
            self._populate(existing)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_rad_ball(self) -> RadiationBall:
        """Build and return a RadiationBall from the current field values."""
        name = self._name_edit.text().strip() or "Radiation Ball"
        cx, cy, cz = self._cx.value(), self._cy.value(), self._cz.value()
        return RadiationBall(
            name=name,
            center=np.array([cx, cy, cz], dtype=float),
            radius=self._radius.value(),
            flux=self._flux.value(),
            active=True,
        )

    def set_centre(self, xyz: np.ndarray) -> None:
        """Fill the centre X/Y/Z fields — called after a viewport pick."""
        self._cx.setValue(float(xyz[0]))
        self._cy.setValue(float(xyz[1]))
        self._cz.setValue(float(xyz[2]))
        self.raise_()
        self.activateWindow()

    def set_pick_mode_active(self, active: bool) -> None:
        """Grey out the pick button while viewport picking is in progress."""
        self._btn_pick_viewport.setEnabled(not active)
        self._btn_use_axis.setEnabled(not active)
        self._btn_pick_viewport.setText(
            "Click in viewport…" if active else "Pick from viewport"
        )

    # ── Qt override ───────────────────────────────────────────────────────────

    def accept(self) -> None:
        try:
            ball = self.get_rad_ball()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))
            return
        self.ball_accepted.emit(ball)
        super().accept()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        name_box = QGroupBox("Source")
        nf = QFormLayout(name_box)
        self._name_edit = QLineEdit("Radiation Ball 1")
        nf.addRow("Name:", self._name_edit)
        layout.addWidget(name_box)

        # Centre position
        pos_box = QGroupBox("Centre position [m]")
        pf = QFormLayout(pos_box)
        self._cx = self._spin(-9999, 9999, 0.0, decimals=3)
        self._cy = self._spin(-9999, 9999, 0.0, decimals=3)
        self._cz = self._spin(-9999, 9999, 0.0, decimals=3)
        pf.addRow("X:", self._cx)
        pf.addRow("Y:", self._cy)
        pf.addRow("Z:", self._cz)

        pick_row = QHBoxLayout()
        self._btn_pick_viewport = QPushButton("Pick from viewport")
        self._btn_pick_viewport.setToolTip(
            "Click a point on the structure in the 3-D viewport to set the centre"
        )
        self._btn_pick_viewport.clicked.connect(self._on_pick_viewport)
        self._btn_use_axis = QPushButton("Use axis origin")
        self._btn_use_axis.setToolTip(
            "Copy the current axis marker position as the ball centre"
        )
        self._btn_use_axis.clicked.connect(self._on_use_axis)
        pick_row.addWidget(self._btn_pick_viewport)
        pick_row.addWidget(self._btn_use_axis)
        pf.addRow(pick_row)
        layout.addWidget(pos_box)

        # Ball radius / flux
        zone_box = QGroupBox("Radiation ball")
        zf = QFormLayout(zone_box)
        self._radius = self._spin(0.01, 9999.0,   5.0,      decimals=2, step=0.5)
        self._flux   = self._spin(0.0,  1.0e9,    350000.0, decimals=0, step=1000.0)
        zf.addRow("Radius [m]:", self._radius)
        zf.addRow("Surface flux  [W/m²]:", self._flux)
        layout.addWidget(zone_box)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_pick_viewport(self) -> None:
        self.set_pick_mode_active(True)
        self.pick_viewport_requested.emit()

    def _on_use_axis(self) -> None:
        self.use_axis_origin_requested.emit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _spin(lo: float, hi: float, val: float,
              decimals: int = 3, step: float = 0.1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        return sb

    def _populate(self, ball: RadiationBall) -> None:
        self._name_edit.setText(ball.name)
        self._cx.setValue(float(ball.center[0]))
        self._cy.setValue(float(ball.center[1]))
        self._cz.setValue(float(ball.center[2]))
        self._radius.setValue(ball.radius)
        self._flux.setValue(ball.flux)
