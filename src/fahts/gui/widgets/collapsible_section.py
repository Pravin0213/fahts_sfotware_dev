"""CollapsibleSection — a titled section that expands / collapses when its header is clicked."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """
    Header button (▸ / ▾ + title) above a content widget that is shown only when expanded.

    Args:
        title:    header text.
        content:  the widget to show / hide (typically a QGroupBox or form widget).
        expanded: initial state.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = content
        self._header = QToolButton(self)
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._header.toggled.connect(self.set_expanded)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._header)
        lay.addWidget(line)
        lay.addWidget(content)
        content.setVisible(expanded)

    @property
    def expanded(self) -> bool:
        return self._header.isChecked()

    def set_expanded(self, on: bool) -> None:
        if self._header.isChecked() != on:
            self._header.setChecked(on)          # re-enters via toggled
            return
        self._header.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self._content.setVisible(on)
        self.toggled.emit(on)
