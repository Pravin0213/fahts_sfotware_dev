"""
LegendRangeDialog — set a custom min/max fringe range for the 3-D temperature legend.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)


class LegendRangeDialog(QDialog):
    """
    Modal dialog for customising the temperature legend fringe range.

    Emits ``range_accepted(lo, hi)`` when the user clicks Apply/OK.
    Emits ``range_reset()`` when the user clicks Reset to Auto.
    """

    range_accepted: pyqtSignal = pyqtSignal(float, float)
    range_reset: pyqtSignal = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        current_lo: float = 20.0,
        current_hi: float = 1000.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Legend Fringe Range")
        self.setModal(True)
        self.setMinimumWidth(280)

        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 4)
        form.setSpacing(8)

        self._sb_lo = QDoubleSpinBox()
        self._sb_lo.setRange(-273.15, 99_999.0)
        self._sb_lo.setDecimals(1)
        self._sb_lo.setSuffix(" °C")
        self._sb_lo.setValue(current_lo)

        self._sb_hi = QDoubleSpinBox()
        self._sb_hi.setRange(-273.15, 99_999.0)
        self._sb_hi.setDecimals(1)
        self._sb_hi.setSuffix(" °C")
        self._sb_hi.setValue(current_hi)

        form.addRow("Min:", self._sb_lo)
        form.addRow("Max:", self._sb_hi)

        btn_reset = QPushButton("Reset to Auto")
        btn_reset.setToolTip("Let the legend range be derived automatically from the data")
        btn_reset.clicked.connect(self._on_reset)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addLayout(form)
        root.addLayout(btn_row)
        root.addWidget(bb)

    def _on_ok(self) -> None:
        lo = self._sb_lo.value()
        hi = self._sb_hi.value()
        if hi <= lo:
            self._sb_hi.setValue(lo + 1.0)
            hi = lo + 1.0
        self.range_accepted.emit(lo, hi)
        self.accept()

    def _on_reset(self) -> None:
        self.range_reset.emit()
        self.accept()
