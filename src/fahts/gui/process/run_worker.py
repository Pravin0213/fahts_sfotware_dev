"""Background thread that runs one ``VesselCase`` (``fahts.coupling.run_case``)."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal

from fahts.coupling import SimulationCancelled, VesselCase, run_case


class CaseRunWorker(QThread):
    """Signals: progress(time_s, t_end_s), finished(CaseResult), error(str), cancelled()."""

    progress = pyqtSignal(float, float)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, case: VesselCase, parent=None) -> None:
        super().__init__(parent)
        self._case = case
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = run_case(self._case, progress=lambda t, T: self.progress.emit(t, T),
                              cancel=self._cancel.is_set)
            self.finished_ok.emit(result)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.error.emit(f"{type(exc).__name__}: {exc}")
