"""Fire section: heat-load time series (background + peak flux) and the peak (jet) zone, with
sketches of the load history and of where the peak zone sits relative to the liquid."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Wedge
from PyQt6.QtWidgets import (QAbstractItemView, QGridLayout, QHeaderView, QLabel, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout)

from fahts.gui.process.fields import dspin, form_group
from fahts.gui.process.forms import _Form

HEADERS = ["Time [s]", "Background [kW/m²]", "Peak [kW/m²]"]


class FireForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        box, f = form_group("Incident heat flux vs time")
        note = QLabel("Incident flux on the shell. The peak flux applies to the peak zone below, "
                      "the background flux to the rest. Linear between rows, constant after the "
                      "last row; set both to 0 for no fire.")
        note.setWordWrap(True)
        f.addRow(note)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(160)
        f.addRow(self.table)
        grid = QGridLayout()
        self.btn_add = QPushButton("Add row")
        self.btn_remove = QPushButton("Remove")
        grid.addWidget(self.btn_add, 0, 0)
        grid.addWidget(self.btn_remove, 0, 1)
        f.addRow(grid)

        qbox, qf = form_group("Quick fill: constant fire")
        self.q_bg = dspin(0.0, 2000.0, 1, "kW/m²")
        self.q_pk = dspin(0.0, 2000.0, 1, "kW/m²")
        self.q_until = dspin(1.0, 1e6, 0, "s")
        self.btn_fill = QPushButton("Replace table")
        qf.addRow("Background flux", self.q_bg)
        qf.addRow("Peak flux", self.q_pk)
        qf.addRow("Until", self.q_until)
        qf.addRow(self.btn_fill)
        f.addRow(qbox)
        lay.addWidget(box)

        box, f = form_group("Peak (jet) fire zone")
        self.xi0 = dspin(0.0, 1.0, 3, tip="Start along the shell, fraction of the length")
        self.xi1 = dspin(0.0, 1.0, 3, tip="End along the shell, fraction of the length")
        self.circ = dspin(0.0, 360.0, 1, "°", tip="Width of the zone around the circumference")
        self.attack = dspin(0.0, 360.0, 1, "°", tip="Centre of the zone, angle from the top "
                                                    "(0 = top, 180 = bottom)")
        f.addRow("Starts at (fraction of length)", self.xi0)
        f.addRow("Ends at (fraction of length)", self.xi1)
        f.addRow("Circumferential width", self.circ)
        f.addRow("Centre angle from top", self.attack)
        self.zone_info = QLabel()
        self.zone_info.setWordWrap(True)
        f.addRow(self.zone_info)
        lay.addWidget(box)

        self.fig = Figure(figsize=(5.5, 4.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(330)
        lay.addWidget(self.canvas)
        lay.addStretch(1)

        self._D, self._L, self._level = 2.0, 6.0, 0.0
        self.btn_add.clicked.connect(self._add_after_last)
        self.btn_remove.clicked.connect(self._remove_rows)
        self.btn_fill.clicked.connect(self._fill_constant)
        self.table.itemChanged.connect(lambda *_: (self._redraw(), self.changed.emit()))
        for w in (self.xi0, self.xi1, self.circ, self.attack):
            w.valueChanged.connect(self._redraw)
        self._watch(self.xi0, self.xi1, self.circ, self.attack)

    # ------------------------------------------------------------------ table
    def _set_rows(self, rows):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            for c, v in enumerate(r):
                self.table.setItem(i, c, QTableWidgetItem(f"{v:g}"))
        self.table.blockSignals(False)
        self._redraw()

    def rows(self) -> list[tuple[float, float, float]]:
        out = []
        for r in range(self.table.rowCount()):
            vals = []
            for c in range(3):
                it = self.table.item(r, c)
                try:
                    vals.append(float(it.text().replace(",", ".")) if it else float("nan"))
                except ValueError:
                    vals.append(float("nan"))
            out.append(tuple(vals))
        return out

    def _add_after_last(self):
        rows = self.rows()
        last = rows[-1] if rows else (0.0, 0.0, 0.0)
        self._set_rows(rows + [(last[0] + 600.0, last[1], last[2])])
        self.changed.emit()

    def _remove_rows(self):
        drop = {i.row() for i in self.table.selectedIndexes()}
        self._set_rows([r for i, r in enumerate(self.rows()) if i not in drop])
        self.changed.emit()

    def _fill_constant(self):
        bg, pk, t = self.q_bg.value(), self.q_pk.value(), self.q_until.value()
        self._set_rows([(0.0, bg, pk), (t, bg, pk), (t + 1.0, 0.0, 0.0)])
        self.changed.emit()

    # ------------------------------------------------------------------ sketches
    def set_geometry(self, D: float, L: float, level: float):
        self._D, self._L, self._level = D, L, level
        self._redraw()

    def _redraw(self):
        rows = [r for r in self.rows() if all(np.isfinite(r))]
        fig = self.fig
        fig.clear()
        ax_q = fig.add_subplot(2, 1, 1)
        if rows:
            a = np.array(rows)
            ax_q.plot(a[:, 0] / 60, a[:, 1], "-o", ms=3, label="background", color="tab:orange")
            ax_q.plot(a[:, 0] / 60, a[:, 2], "--s", ms=3, label="peak", color="tab:red")
            ax_q.legend(fontsize=8, loc="best")
        ax_q.set_xlabel("time [min]", fontsize=8)
        ax_q.set_ylabel("flux [kW/m²]", fontsize=8)
        ax_q.tick_params(labelsize=8)
        ax_q.grid(alpha=0.3)

        # end view: circle, liquid, peak arc (angle from the top, drawn clockwise)
        ax_e = fig.add_subplot(2, 2, 3)
        R, lev = 1.0, min(self._level / max(self._D, 1e-9), 1.0) * 2.0
        ax_e.add_patch(Circle((0, 0), R, fill=False, lw=1.5, color="0.5"))
        if lev > 0:
            # liquid below y = -R + lev: the arc +-phi around the bottom, closed by the chord
            phi = np.arccos(np.clip(1.0 - lev, -1.0, 1.0))
            t = np.linspace(-np.pi / 2 - phi, -np.pi / 2 + phi, 100)
            ax_e.fill(np.cos(t) * R, np.sin(t) * R, color="tab:blue", alpha=0.35, lw=0)
        c, w = self.attack.value(), self.circ.value()
        if w > 0:
            # matplotlib wedge angles: counter-clockwise from +x; "from top clockwise" -> 90 - a
            ax_e.add_patch(Wedge((0, 0), R * 1.12, 90 - (c + w / 2), 90 - (c - w / 2),
                                 width=0.12, color="tab:red"))
        ax_e.set_xlim(-1.25, 1.25)
        ax_e.set_ylim(-1.25, 1.25)
        ax_e.set_aspect("equal")
        ax_e.axis("off")
        ax_e.set_title("end view", fontsize=8)

        # side view: shell, liquid, peak zone along the length
        ax_s = fig.add_subplot(2, 2, 4)
        ax_s.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.5, color="0.5"))
        if lev > 0:
            ax_s.add_patch(Rectangle((0, 0), 1, lev / 2.0, color="tab:blue", alpha=0.35, lw=0))
        x0, x1 = self.xi0.value(), self.xi1.value()
        if x1 > x0 and w > 0:
            ax_s.add_patch(Rectangle((x0, -0.08), x1 - x0, 1.16, fill=False, lw=2,
                                     color="tab:red"))
        ax_s.set_xlim(-0.05, 1.05)
        ax_s.set_ylim(-0.15, 1.15)
        ax_s.axis("off")
        ax_s.set_title(f"side view (L = {self._L:g} m)", fontsize=8)
        self.canvas.draw_idle()

        area = max(x1 - x0, 0.0) * w / 360.0
        pk_differs = rows and any(r[1] != r[2] for r in rows)
        self.zone_info.setText(
            f"Peak zone covers {100 * area:.2f} % of the shell. "
            + ("" if pk_differs else "Peak flux equals background: the zone has no effect."))

    # ------------------------------------------------------------------ case <-> widgets
    def set_case(self, case):
        h = case.heat_load
        self.xi0.setValue(h.peak.xi_start)
        self.xi1.setValue(h.peak.xi_end)
        self.circ.setValue(h.peak.circ_deg)
        self.attack.setValue(h.peak.attack_deg)
        self.q_until.setValue(case.run.t_end_s)
        self._set_rows(list(zip(h.times_s, h.q_background_kW_m2, h.q_peak_kW_m2)))
        c = case.contents
        self.set_geometry(case.vessel.inner_diameter_m, case.vessel.length_m,
                          c.hc_liquid_depth_m + c.water_depth_m)

    def apply(self, case):
        h = case.heat_load
        rows = self.rows()
        h.times_s = [r[0] for r in rows]
        h.q_background_kW_m2 = [r[1] for r in rows]
        h.q_peak_kW_m2 = [r[2] for r in rows]
        h.peak.xi_start = self.xi0.value()
        h.peak.xi_end = self.xi1.value()
        h.peak.circ_deg = self.circ.value()
        h.peak.attack_deg = self.attack.value()
