"""
Phase 2.3 — FireZoneDialog
Qt dialog for placing / editing a rectangular fire zone in 3-D space.

The dialog is non-modal by design — callers should use show() so the user
can freely move it while interacting with the 3-D viewport.

Signals
-------
zone_accepted(object)          : emitted on OK with the built FireZone
pick_viewport_requested()      : user wants to click the viewport to pick centre
use_axis_origin_requested()    : user wants to use the current axis marker position
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fahts.core.heat.sources.fire_zone import FireCurve, FireCurveType, FireZone


class FireZoneDialog(QDialog):
    """
    Non-modal dialog for creating or editing a FireZone.

    Usage (non-blocking)
    --------------------
    dlg = FireZoneDialog(parent=self)
    dlg.zone_accepted.connect(my_handler)          # receives FireZone
    dlg.pick_viewport_requested.connect(on_pick)   # enable viewport picking
    dlg.use_axis_origin_requested.connect(on_axis) # snap to axis marker
    dlg.show()
    """

    zone_accepted: pyqtSignal = pyqtSignal(object)       # emits FireZone
    pick_viewport_requested: pyqtSignal = pyqtSignal()
    use_axis_origin_requested: pyqtSignal = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        existing: FireZone | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Fire Zone" if existing is None else "Edit Fire Zone")
        self.setMinimumWidth(440)
        # Non-modal: window can be freely moved while viewport stays interactive
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._build_ui()
        if existing is not None:
            self._populate(existing)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fire_zone(self) -> FireZone:
        """Build and return a FireZone from the current field values."""
        name   = self._name_edit.text().strip() or "Fire Zone"
        cx, cy, cz = self._cx.value(), self._cy.value(), self._cz.value()
        dx, dy, dz = self._dx.value(), self._dy.value(), self._dz.value()
        eps    = self._eps.value()
        h_conv = self._hconv.value()

        curve_type = FireCurveType(self._curve_combo.currentData())
        curve = FireCurve(
            curve_type=curve_type,
            user_points=self._parse_user_points(),
        )
        return FireZone(
            name=name,
            center=np.array([cx, cy, cz], dtype=float),
            dims=np.array([dx, dy, dz], dtype=float),
            curve=curve,
            epsilon_fire=eps,
            h_conv=h_conv,
            active=True,
        )

    def set_centre(self, xyz: np.ndarray) -> None:
        """
        Fill the centre X/Y/Z fields with the given world coordinates.

        Called externally after a viewport pick or axis-origin snap.
        """
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        self._cx.setValue(x)
        self._cy.setValue(y)
        self._cz.setValue(z)
        # Bring dialog to front so user can see the update
        self.raise_()
        self.activateWindow()

    def set_pick_mode_active(self, active: bool) -> None:
        """Grey out the dialog while viewport picking is in progress."""
        self._btn_pick_viewport.setEnabled(not active)
        self._btn_use_axis.setEnabled(not active)
        if active:
            self._btn_pick_viewport.setText("Click in viewport…")
        else:
            self._btn_pick_viewport.setText("Pick from viewport")

    # ── Qt override ───────────────────────────────────────────────────────────

    def accept(self) -> None:
        zone = self.get_fire_zone()
        self.zone_accepted.emit(zone)
        super().accept()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        name_box = QGroupBox("Zone")
        nf = QFormLayout(name_box)
        self._name_edit = QLineEdit("Fire Zone 1")
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

        # Pick-centre row: "Pick from viewport" and "Use Axis Origin"
        pick_row = QHBoxLayout()
        self._btn_pick_viewport = QPushButton("Pick from viewport")
        self._btn_pick_viewport.setToolTip(
            "Click a point on the structure in the 3-D viewport to set the centre"
        )
        self._btn_pick_viewport.clicked.connect(self._on_pick_viewport)

        self._btn_use_axis = QPushButton("Use axis origin")
        self._btn_use_axis.setToolTip(
            "Copy the current axis marker position as the fire zone centre\n"
            "(enable the axis marker via View → Axis Marker)"
        )
        self._btn_use_axis.clicked.connect(self._on_use_axis)

        pick_row.addWidget(self._btn_pick_viewport)
        pick_row.addWidget(self._btn_use_axis)
        pf.addRow(pick_row)
        layout.addWidget(pos_box)

        # Dimensions
        dim_box = QGroupBox("Full extents [m]")
        df = QFormLayout(dim_box)
        self._dx = self._spin(0.01, 9999, 5.0, decimals=3)
        self._dy = self._spin(0.01, 9999, 5.0, decimals=3)
        self._dz = self._spin(0.01, 9999, 5.0, decimals=3)
        df.addRow("Width  (X):", self._dx)
        df.addRow("Depth  (Y):", self._dy)
        df.addRow("Height (Z):", self._dz)
        layout.addWidget(dim_box)

        # Fire curve
        curve_box = QGroupBox("Fire curve")
        cf = QFormLayout(curve_box)
        self._curve_combo = QComboBox()
        self._curve_combo.addItem("ISO 834 (Standard cellulosic)",
                                  FireCurveType.ISO_834.value)
        self._curve_combo.addItem("Hydrocarbon (EN 1991-1-2)",
                                  FireCurveType.HYDROCARBON.value)
        self._curve_combo.addItem("User-defined (piecewise linear)",
                                  FireCurveType.USER_DEFINED.value)
        self._curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        cf.addRow("Curve:", self._curve_combo)

        self._user_hint = QLabel(
            "One (t [s], T [°C]) pair per line, comma-separated.\n"
            "Example:  0, 20\n  60, 500\n  3600, 1000"
        )
        self._user_hint.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._user_edit = QPlainTextEdit()
        self._user_edit.setFixedHeight(80)
        self._user_edit.setPlaceholderText("0, 20\n60, 500\n3600, 1000")
        self._user_hint.setVisible(False)
        self._user_edit.setVisible(False)
        cf.addRow(self._user_hint)
        cf.addRow("Points:", self._user_edit)
        layout.addWidget(curve_box)

        # Physical parameters
        phys_box = QGroupBox("Physical parameters")
        pf2 = QFormLayout(phys_box)
        self._eps   = self._spin(0.0, 1.0, 1.0, decimals=2, step=0.05)
        self._hconv = self._spin(0.0, 200.0, 25.0, decimals=1, step=5.0)
        pf2.addRow("Fire emissivity ε:", self._eps)
        pf2.addRow("h_conv [W/(m²·K)]:", self._hconv)
        layout.addWidget(phys_box)

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

    def _on_curve_changed(self, _: int) -> None:
        is_user = (
            self._curve_combo.currentData() == FireCurveType.USER_DEFINED.value
        )
        self._user_hint.setVisible(is_user)
        self._user_edit.setVisible(is_user)
        self.adjustSize()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _spin(
        lo: float, hi: float, val: float,
        decimals: int = 3, step: float = 0.1,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        return sb

    def _parse_user_points(self) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for line in self._user_edit.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                result.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
        return result

    def _populate(self, zone: FireZone) -> None:
        self._name_edit.setText(zone.name)
        self._cx.setValue(float(zone.center[0]))
        self._cy.setValue(float(zone.center[1]))
        self._cz.setValue(float(zone.center[2]))
        self._dx.setValue(float(zone.dims[0]))
        self._dy.setValue(float(zone.dims[1]))
        self._dz.setValue(float(zone.dims[2]))
        self._eps.setValue(zone.epsilon_fire)
        self._hconv.setValue(zone.h_conv)
        idx = self._curve_combo.findData(zone.curve.curve_type.value)
        if idx >= 0:
            self._curve_combo.setCurrentIndex(idx)
        if zone.curve.user_points:
            lines = "\n".join(f"{t}, {T}" for t, T in zone.curve.user_points)
            self._user_edit.setPlainText(lines)
