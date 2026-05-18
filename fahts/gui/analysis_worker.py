"""
Task 3.8 — AnalysisWorker: Qt thread wrapper around run_analysis.

Signals
-------
progress(current: int, total: int, eid: int)
    Emitted before each element is solved.
    current = 0-based index; total = total number of elements.
    eid = element ID being solved (-1 on the final "done" tick).

finished(result: TemperatureField)
    Emitted when all elements have been solved successfully.

error(message: str)
    Emitted when an unhandled exception stops the run.

cancelled()
    Emitted when the user cancels via cancel().
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal

from fahts.core.heat.solver.analysis_runner import AnalysisCancelledError, run_analysis
from fahts.core.heat.sources.fire_zone import FireZone
from fahts.core.model.fem_model import FEMModel
from fahts.core.results.analysis_config import AnalysisConfig


class AnalysisWorker(QThread):
    """
    Runs the transient heat-transfer analysis in a background thread.

    Usage
    -----
        worker = AnalysisWorker(model, fire_zones, config)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.cancelled.connect(on_cancelled)
        worker.start()
        # later:
        worker.cancel()
    """

    progress:  pyqtSignal = pyqtSignal(int, int, int)   # current, total, eid
    finished:  pyqtSignal = pyqtSignal(object)           # TemperatureField
    error:     pyqtSignal = pyqtSignal(str)
    cancelled: pyqtSignal = pyqtSignal()

    def __init__(
        self,
        model: FEMModel,
        fire_zones: list[FireZone],
        config: AnalysisConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._fire_zones = fire_zones
        self._config = config
        self._cancel_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Request cancellation.  The worker will stop before the next element."""
        self._cancel_event.set()

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the analysis.  Called by QThread.start() in a worker thread."""
        try:
            result = run_analysis(
                model=self._model,
                fire_zones=self._fire_zones,
                config=self._config,
                progress_cb=lambda cur, tot, eid: self.progress.emit(cur, tot, eid),
                cancel_check=self._cancel_event.is_set,
            )
            self.finished.emit(result)
        except AnalysisCancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
