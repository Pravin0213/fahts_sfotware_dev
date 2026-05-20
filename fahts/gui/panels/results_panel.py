"""
Phase 4.5 / 4.6 — results_panel.py
Left-sidebar panel with two complementary result views:
  • 2-D cross-section temperature contour (BOX sections, inferno, gouraud) — Task 4.5
  • Peak-element temperature-time graph with draggable step slider        — Task 4.6
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)

from fahts.core.model.fem_model import FEMModel
from fahts.core.model.section import BoxSection
from fahts.core.heat.section_mesh.box_mesher import BoxMesher

_CRIT_T = 600.0   # EN 1993-1-2 critical steel temperature [°C]


class ResultsPanel(QWidget):
    """
    Sidebar panel with two analysis result views.

    Public API
    ----------
    show_results(result)
        Rebuild the global hottest-element T-t graph.  Call once after analysis.
    show_section(eid, model, result, t_idx=0)
        Display / refresh the cross-section contour for *eid* at step *t_idx*.
    update_time(t_idx)
        Move the T-t marker to step *t_idx* and redraw cross-section.
    clear()
        Reset all plots to placeholder state.

    Signals
    -------
    time_hovered(int)
        Emitted when the slider is dragged; carries the time-step index so the
        3-D view can update in real time.
    """

    time_hovered: pyqtSignal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._eid: int | None = None
        self._result: object = None
        self._t_idx: int = 0
        self._hottest_col: int | None = None
        self._mesh_nodes: np.ndarray | None = None
        self._triang: mtri.Triangulation | None = None
        self._tt_marker: Line2D | None = None   # red × on the curve at current step
        self._T_hist: np.ndarray | None = None  # T history of the hottest element
        self._tt_ax = None
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def show_results(self, result: object) -> None:
        """Rebuild the global T-t graph for the hottest element in *result*."""
        self._result = result
        self._t_idx = 0
        self._redraw_tt()

    def show_section(
        self,
        eid: int,
        model: FEMModel,
        result: object,
        t_idx: int = 0,
    ) -> None:
        """Display the cross-section contour for *eid* and update the T-t graph."""
        self._eid = eid
        self._result = result
        self._t_idx = t_idx
        self._build_mesh(eid, model)
        self._redraw()
        self._redraw_tt()

    def update_time(self, t_idx: int) -> None:
        """Move the T-t marker and refresh the cross-section for step *t_idx*."""
        self._t_idx = t_idx
        if self._eid is not None and self._result is not None:
            self._redraw()
        self._update_tt_marker()

    def clear(self) -> None:
        """Reset all plots to placeholder state."""
        self._eid = None
        self._result = None
        self._hottest_col = None
        self._T_hist = None
        self._mesh_nodes = None
        self._triang = None
        self._tt_marker = None
        self._tt_ax = None
        self._placeholder.show()
        self._canvas.hide()
        self._tt_placeholder.show()
        self._tt_canvas.hide()
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, 0)
        self._time_slider.setEnabled(False)
        self._time_slider.blockSignals(False)
        self._step_label.setText("—")

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── Cross-section block ───────────────────────────────────────────────
        hdr_cs = QLabel("Cross-section")
        hdr_cs.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        layout.addWidget(hdr_cs)

        self._placeholder = QLabel(
            "Select a beam with\nanalysis results\nto view cross-section."
        )
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._placeholder.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self._placeholder)

        self._fig = Figure(figsize=(2.5, 2.2), tight_layout=True)
        self._fig.patch.set_facecolor("#2b2b2b")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.hide()
        layout.addWidget(self._canvas, 1)

        # ── Temperature-time block ────────────────────────────────────────────
        hdr_tt = QLabel("Peak Temperature vs Time")
        hdr_tt.setStyleSheet("font-weight: bold; color: #aaaaaa; margin-top: 4px;")
        layout.addWidget(hdr_tt)

        self._tt_placeholder = QLabel(
            "Run analysis to view\ntemperature history.\n\nDrag the slider below\n"
            "to step through time."
        )
        self._tt_placeholder.setWordWrap(True)
        self._tt_placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._tt_placeholder.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self._tt_placeholder)

        self._tt_fig = Figure(figsize=(2.5, 2.0), tight_layout=True)
        self._tt_fig.patch.set_facecolor("#f5f0e8")
        self._tt_canvas = FigureCanvas(self._tt_fig)
        self._tt_canvas.hide()
        layout.addWidget(self._tt_canvas, 1)

        # ── Slider row (below graph) ──────────────────────────────────────────
        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.setEnabled(False)
        self._time_slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            "  height: 4px; background: #cccccc; border-radius: 2px;"
            "}"
            "QSlider::handle:horizontal {"
            "  width: 14px; height: 14px;"
            "  background: #cc1111; border-radius: 7px;"
            "  margin: -5px 0;"
            "}"
            "QSlider::handle:horizontal:disabled { background: #999999; }"
        )
        self._step_label = QLabel("—")
        self._step_label.setStyleSheet("color: #888888; font-size: 8px; min-width: 30px;")
        self._step_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(2, 0, 2, 2)
        slider_row.setSpacing(4)
        slider_row.addWidget(self._time_slider)
        slider_row.addWidget(self._step_label)
        layout.addLayout(slider_row)

        self._time_slider.valueChanged.connect(self._on_slider_changed)

    # ── Mesh building ─────────────────────────────────────────────────────────

    def _build_mesh(self, eid: int, model: FEMModel) -> None:
        """Build Triangulation from the BOX cross-section mesh for *eid*."""
        self._mesh_nodes = None
        self._triang = None

        elem = model.elements.get(eid)
        if elem is None:
            return
        section = model.sections.get(elem.geom_id)
        if not isinstance(section, BoxSection):
            return

        mesh = BoxMesher(section).build()
        nodes = mesh.nodes   # (n_nodes, 2) [y, z] in local cross-section plane
        self._mesh_nodes = nodes

        triangles: list[tuple[int, int, int]] = []
        for q in mesh.quads:
            a, b, c, d = int(q[0]), int(q[1]), int(q[2]), int(q[3])
            triangles.append((a, b, c))
            triangles.append((a, c, d))
        self._triang = mtri.Triangulation(
            nodes[:, 0], nodes[:, 1], np.array(triangles, dtype=np.intp)
        )

    # ── Cross-section plot ────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """Full redraw of the 2-D cross-section temperature plot."""
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")

        if self._triang is None or self._result is None or self._eid is None:
            self._placeholder.show()
            self._canvas.hide()
            return

        T_section: dict = getattr(self._result, "T_section", {})
        if self._eid not in T_section:
            self._placeholder.show()
            self._canvas.hide()
            return

        T_mat = T_section[self._eid]          # (n_steps, n_nodes)
        t_idx = min(self._t_idx, T_mat.shape[0] - 1)
        T_nodes = T_mat[t_idx]                # (n_nodes,)

        T_lo = float(T_nodes.min())
        T_hi = float(T_nodes.max())
        if T_hi - T_lo < 1.0:
            T_hi = T_lo + 1.0

        tc = ax.tripcolor(
            self._triang, T_nodes,
            cmap="inferno", vmin=T_lo, vmax=T_hi, shading="gouraud",
        )
        cb = self._fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("°C", color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

        times = getattr(self._result, "times", None)
        if times is not None:
            t_val = float(times[t_idx])
            t_str = f"{t_val:.0f} s" if t_val < 60 else f"{t_val / 60:.1f} min"
        else:
            t_str = f"step {t_idx}"

        ax.set_title(f"EID {self._eid}  |  {t_str}", color="white", fontsize=8)
        ax.set_xlabel("y [m]", color="white", fontsize=7)
        ax.set_ylabel("z [m]", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

        self._placeholder.hide()
        self._canvas.show()
        self._canvas.draw_idle()

    # ── Temperature-time plot ─────────────────────────────────────────────────

    def _redraw_tt(self) -> None:
        """Full redraw of the global peak-element T-t graph."""
        self._tt_fig.clf()
        _BG = "#f5f0e8"
        self._tt_fig.patch.set_facecolor(_BG)
        ax = self._tt_fig.add_subplot(111)
        ax.set_facecolor(_BG)
        self._tt_ax = ax
        self._tt_marker = None
        self._T_hist = None

        if self._result is None:
            self._tt_placeholder.show()
            self._tt_canvas.hide()
            return

        T_centroid = getattr(self._result, "T_centroid", None)
        times = getattr(self._result, "times", None)
        if T_centroid is None or times is None or T_centroid.shape[1] == 0:
            self._tt_placeholder.show()
            self._tt_canvas.hide()
            return

        # Hottest element — NaN-safe (failed elements produce NaN)
        peak_per_col = np.nanmax(T_centroid, axis=0)
        self._hottest_col = int(np.nanargmax(peak_per_col))
        times_arr = np.asarray(times)
        T_hist = T_centroid[:, self._hottest_col]
        self._T_hist = T_hist

        T_max = float(np.nanmax(T_hist))
        y_top = max(T_max * 1.08, T_max + 50.0)

        element_ids = list(getattr(self._result, "element_ids", []))
        eid_label = (
            element_ids[self._hottest_col]
            if self._hottest_col < len(element_ids)
            else "?"
        )

        # Temperature curve
        ax.plot(times_arr, T_hist, color="black", linewidth=1.8, zorder=2)

        # Red × at the current time step (moves as slider is dragged)
        t_idx = min(self._t_idx, len(times_arr) - 1)
        marker_x = float(times_arr[t_idx])
        T_at_idx = float(T_hist[t_idx]) if not np.isnan(T_hist[t_idx]) else 0.0
        self._tt_marker, = ax.plot(
            marker_x, T_at_idx,
            marker="x", color="red", markersize=10, markeredgewidth=2.0,
            linestyle="none", zorder=5,
        )

        if _CRIT_T <= y_top:
            ax.axhline(
                _CRIT_T, color="#cc8800", linewidth=0.9,
                linestyle="--", alpha=0.7,
            )

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6, color="#aaaaaa")
        ax.set_axisbelow(True)
        ax.set_xlabel("Time", color="black", fontsize=8)
        ax.set_ylabel("Temperature", color="black", fontsize=8)
        ax.set_title(f"Hottest: EID {eid_label}", color="black", fontsize=9)
        ax.tick_params(colors="black", labelsize=7)
        ax.set_xlim(0.0, float(times_arr[-1]))
        ax.set_ylim(0.0, y_top)
        for sp in ax.spines.values():
            sp.set_edgecolor("#888888")
            sp.set_linewidth(0.8)

        # Configure the slider
        n = len(times_arr)
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, n - 1)
        self._time_slider.setValue(t_idx)
        self._time_slider.setEnabled(True)
        self._time_slider.blockSignals(False)
        self._step_label.setText(f"{t_idx + 1}/{n}")

        self._tt_placeholder.hide()
        self._tt_canvas.show()
        self._tt_canvas.draw_idle()

    def _update_tt_marker(self) -> None:
        """Move the red × to the current step without a full graph redraw."""
        if self._tt_marker is None or self._T_hist is None or self._result is None:
            return
        times = np.asarray(getattr(self._result, "times", []))
        if len(times) == 0:
            return
        idx = min(self._t_idx, len(times) - 1)
        t_cur = float(times[idx])
        T_cur = float(self._T_hist[idx]) if not np.isnan(self._T_hist[idx]) else 0.0
        self._tt_marker.set_xdata([t_cur])
        self._tt_marker.set_ydata([T_cur])
        self._tt_canvas.draw_idle()

        self._time_slider.blockSignals(True)
        self._time_slider.setValue(idx)
        self._time_slider.blockSignals(False)
        self._step_label.setText(f"{idx + 1}/{len(times)}")

    # ── Slider interaction ────────────────────────────────────────────────────

    def _on_slider_changed(self, idx: int) -> None:
        """Slider dragged to *idx*: move cursor, refresh cross-section, drive 3-D."""
        self._t_idx = idx
        self._update_tt_marker()
        if self._eid is not None and self._result is not None:
            self._redraw()
        self.time_hovered.emit(idx)
