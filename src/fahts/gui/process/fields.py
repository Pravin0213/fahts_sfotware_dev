"""Small input-field helpers for the process forms (engineering units, fixed decimals)."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QWidget


class ExactDoubleSpinBox(QDoubleSpinBox):
    """A spin box that returns the exact value it was given, not the display-rounded one.

    QDoubleSpinBox rounds to its display decimals, so loading 1.01325 into a 4-decimal box and
    reading it back gives 1.0132: case files and imported decks would silently lose precision.
    Here ``value()`` returns the value passed to ``setValue`` unless the user has changed the
    number (then the edited value is returned).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._exact: float | None = None
        self._setting = False
        self.valueChanged.connect(self._on_value_changed)

    def setValue(self, v: float) -> None:  # noqa: N802 - Qt API
        self._setting = True
        try:
            self._exact = float(v)
            super().setValue(v)
        finally:
            self._setting = False

    def _on_value_changed(self, _v: float) -> None:
        if not self._setting:  # changed by the user (typing, arrows, wheel)
            self._exact = None

    def value(self) -> float:
        return self._exact if self._exact is not None else super().value()


def dspin(lo: float, hi: float, decimals: int = 3, suffix: str = "", step: float | None = None,
          tip: str = "") -> QDoubleSpinBox:
    w = ExactDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setKeyboardTracking(False)
    if suffix:
        w.setSuffix(f" {suffix}")
    w.setSingleStep(step if step is not None else 10 ** -max(decimals - 1, 0))
    if tip:
        w.setToolTip(tip)
    return w


def combo(items: list[str], tip: str = "") -> QComboBox:
    w = QComboBox()
    w.addItems(items)
    if tip:
        w.setToolTip(tip)
    return w


def form_group(title: str, parent: QWidget | None = None, checkable: bool = False
               ) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title, parent)
    box.setCheckable(checkable)
    lay = QFormLayout(box)
    lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return box, lay
