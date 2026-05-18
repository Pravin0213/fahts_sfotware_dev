"""
Phase 4.5 / 4.6 — results_panel.py
Left-sidebar panel with two complementary result views:
  • 2-D cross-section temperature contour (BOX sections, inferno, gouraud) — Task 4.5
  • Peak-element temperature-time graph with hover navigation          — Task 4.6
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

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
        Move the T-t graph cursor to step *t_idx* and redraw cross-section.
    clear()
        Reset all plots to placeholder state.

    Signals
    -------
    time_hovered(int)
        Emitted while the user moves the pointer over the T-t graph; carries
        the nearest time-step index so the 3-D view can update in real time.
    """

    time_hovered: pyqtSignal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._eid: int | None = None
        self._result: object = None
        self._t_idx: int = 0
        self._hottest_col: int | None = None       # column index of hottest element
        self._mesh_nodes: np.ndarray | None = None
        self._triang: mtri.Triangulation | None = None
        self._tt_vline: Line2D | None = None
        self._tt_ax = None                         # axes for the T-t graph
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def show_results(self, result: object) -> None:
        """Rebuild the global T-t graph for the hottest element in *result*."""
        self._result = result
        T_centroid = getattr(result, "T_centroid", None)
        if T_centroid is not None and T_centroid.shape[1] > 0:
            self._hottest_col = int(np.argmax(T_centroid.max(axis=0)))
        else:
            self._hottest_col = None
        self._t_idx = 0
        self._redraw_tt()

    def show_section(
        self,
        eid: int,
        model: FEMModel,
        result: object,
        t_idx: int = 0,
    ) -> None:
        """Display the cross-section contour for element *eid* at step *t_idx*."""
        self._eid = eid
        self._result = result
        self._t_idx = t_idx
        self._build_mesh(eid, model)
        self._redraw()

    def update_time(self, t_idx: int) -> None:
        """Move the T-t graph cursor and refresh the cross-section for step *t_idx*."""
        self._t_idx = t_idx
        if self._eid is not None and self._result is not None:
            self._redraw()
        self._update_tt_marker()

    def clear(self) -> None:
        """Reset all plots to placeholder state."""
        self._eid = None
        self._result = None
        self._hottest_col = None
        self._mesh_nodes = None
        self._triang = None
        self._tt_vline = None
        self._tt_ax = None
        self._placeholder.show()
        self._canvas.hide()
        self._tt_placeholder.show()
        self._tt_canvas.hide()

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
            "Run analysis to view\ntemperature history.\n\nHover over the graph\n"
            "to update the 3-D view."
        )
        self._tt_placeholder.setWordWrap(True)
        self._tt_placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._tt_placeholder.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self._tt_placeholder)

        self._tt_fig = Figure(figsize=(2.5, 2.0), tight_layout=True)
        self._tt_fig.patch.set_facecolor("#2b2b2b")
        self._tt_canvas = FigureCanvas(self._tt_fig)
        self._tt_canvas.hide()
        layout.addWidget(self._tt_canvas, 1)

        # Hover tracking: motion over the T-t graph drives the 3-D time step
        self._tt_canvas.mpl_connect("motion_notify_event", self._on_tt_hover)

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
        ax = self._tt_fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        self._tt_ax = ax
        self._tt_vline = None

        if self._result is None or self._hottest_col is None:
            self._tt_placeholder.show()
            self._tt_canvas.hide()
            return

        T_centroid = getattr(self._result, "T_centroid", None)
        times = getattr(self._result, "times", None)
        if T_centroid is None or times is None:
            self._tt_placeholder.show()
            self._tt_canvas.hide()
            return

        times_arr = np.asarray(times)
        T_hist = T_centroid[:, self._hottest_col]   # (n_steps,)

        # Global temperature range across all elements and all time steps
        T_global_min = float(np.nanmin(T_centroid))
        T_global_max = float(np.nanmax(T_centroid))
        y_pad = max((T_global_max - T_global_min) * 0.05, 5.0)

        element_ids = list(getattr(self._result, "element_ids", []))
        eid_label = (
            element_ids[self._hottest_col]
            if self._hottest_col < len(element_ids)
            else "?"
        )

        ax.plot(times_arr, T_hist, color="#FF0000", linewidth=2.0)

        if _CRIT_T <= T_global_max + y_pad:
            ax.axhline(
                _CRIT_T, color="#ffcc00", linewidth=0.9,
                linestyle="--", alpha=0.85, label="600 °C",
            )

        t_cur = float(times_arr[min(self._t_idx, len(times_arr) - 1)])
        self._tt_vline = ax.axvline(
            t_cur, color="#ffffff", linewidth=1.2, linestyle=":", alpha=0.9
        )

        ax.set_xlabel("Time [s]", color="white", fontsize=7)
        ax.set_ylabel("T [°C]", color="white", fontsize=7)
        ax.set_title(f"Hottest: EID {eid_label}", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        ax.set_xlim(float(times_arr[0]), float(times_arr[-1]))
        ax.set_ylim(T_global_min - y_pad, T_global_max + y_pad)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

        self._tt_placeholder.hide()
        self._tt_canvas.show()
        self._tt_canvas.draw_idle()

    def _update_tt_marker(self) -> None:
        """Move the cursor line on the T-t graph without full redraw."""
        if self._tt_vline is None or self._result is None:
            return
        times = np.asarray(self._result.times)
        t_cur = float(times[min(self._t_idx, len(times) - 1)])
        self._tt_vline.set_xdata([t_cur, t_cur])
        self._tt_canvas.draw_idle()

    def _on_tt_hover(self, event) -> None:
        """Matplotlib motion callback: snap cursor to nearest step, emit signal."""
        if self._tt_ax is None or event.inaxes is not self._tt_ax:
            return
        if self._result is None or event.xdata is None:
            return
        times = np.asarray(self._result.times)
        idx = int(np.argmin(np.abs(times - event.xdata)))
        # Move cursor immediately for snappy feel before the signal roundtrip
        if self._tt_vline is not None:
            self._tt_vline.set_xdata([float(times[idx]), float(times[idx])])
            self._tt_canvas.draw_idle()
        self.time_hovered.emit(idx)
