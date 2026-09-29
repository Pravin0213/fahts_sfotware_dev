"""
USFOS-style output console for FAHTS analysis progress.

Replaces QProgressDialog with a scrollable monospace console that prints
analysis parameters, per-element results, and a final summary in the same
format as the USFOS FAHTS .out file.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class AnalysisConsoleDialog(QDialog):
    """
    Scrollable console dialog shown during heat-transfer analysis.

    The caller should:
      1. Connect worker.log_line  → append_line
      2. Connect worker.progress  → set_progress
      3. Connect worker.finished  → set_complete  (passing the result)
      4. Connect worker.error     → set_failed
      5. Connect worker.cancelled → set_cancelled
      6. Connect rejected         → worker.cancel   (Cancel button / window X)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FAHTS — Heat Transfer Analysis")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(720, 560)
        self.setMinimumSize(560, 380)
        self._finished = False
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        font = QFont("Courier New", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #f5f5f0;"
            "  color: #111111;"
            "  border: 1px solid #b0b0b0;"
            "}"
        )
        layout.addWidget(self._text, stretch=1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("  Initialising …")
        layout.addWidget(self._progress)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn = QPushButton("Cancel")
        self._btn.setFixedWidth(100)
        self._btn.clicked.connect(self._on_btn_clicked)
        btn_row.addWidget(self._btn)
        layout.addLayout(btn_row)

    # ── Public slots ──────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def append_line(self, text: str) -> None:
        """Append one line of USFOS-style text and scroll to bottom."""
        self._text.appendPlainText(text)
        self._text.moveCursor(QTextCursor.MoveOperation.End)

    @pyqtSlot(int, int, int)
    def set_progress(self, current: int, total: int, eid: int) -> None:
        """Update progress bar; eid=-1 means time-step progress."""
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
            if eid >= 0:
                self._progress.setFormat(f"  Element {current} of {total}  (ID {eid})")
            else:
                pct = int(100 * current / total) if total else 0
                self._progress.setFormat(f"  Step {current} of {total}  ({pct}%)")

    def set_complete(self) -> None:
        """Mark analysis done; change Cancel → Close."""
        self._finished = True
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("  Complete")
        self._btn.setText("Close")

    def set_failed(self, message: str) -> None:
        """Show error state."""
        self._finished = True
        self._progress.setFormat("  Failed")
        self._btn.setText("Close")
        self.append_line("")
        self.append_line("     *** ERROR ***")
        self.append_line(f"     {message}")

    def set_cancelled(self) -> None:
        """Acknowledge user cancellation."""
        self._finished = True
        self._progress.setFormat("  Cancelled")
        self._btn.setText("Close")
        self.append_line("")
        self.append_line("     Analysis cancelled by user.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_btn_clicked(self) -> None:
        if self._finished:
            self.accept()
        else:
            self.reject()      # connected to worker.cancel in main_window

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._finished:
            self.reject()
            event.ignore()
        else:
            event.accept()
