"""Results of a vessel-in-fire run: summary, plots and failure-time table, export."""

from __future__ import annotations

import math

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from fahts.coupling.report import REGIONS, regions_with_area, summary_lines
from fahts.coupling.runner import CaseResult

class ResultsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: CaseResult | None = None
        self._plot_pending = False
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("<b>Results</b>")
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_xlsx = QPushButton("Export Excel…")
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.btn_csv)
        top.addWidget(self.btn_xlsx)
        lay.addLayout(top)
        self.summary = QTableWidget(0, 2)
        self.summary.horizontalHeader().setVisible(False)
        self.summary.verticalHeader().setVisible(False)
        self.summary.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.summary.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.summary.setMaximumHeight(300)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.figures = {}
        for name in ("Pressure", "Temperatures", "Inventory", "Release", "Stress"):
            fig = Figure(figsize=(7, 4.5), tight_layout=True)
            canvas = FigureCanvasQTAgg(fig)
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.addWidget(NavigationToolbar2QT(canvas, page))
            pl.addWidget(canvas, 1)
            self.tabs.addTab(page, name)
            self.figures[name] = (fig, canvas)
        self.fail_table = QTableWidget()
        self.tabs.addTab(self.fail_table, "Failure times")
        lay.addWidget(self.tabs, 1)
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_xlsx.clicked.connect(self._export_xlsx)
        self.btn_csv.setEnabled(False)
        self.btn_xlsx.setEnabled(False)

    # ------------------------------------------------------------------ show
    def show_result(self, res: CaseResult) -> None:
        self.result = res
        self.title.setText(f"<b>Results — {res.case.name}</b>")
        rows = summary_lines(res)
        self.summary.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.summary.setItem(i, 0, QTableWidgetItem(k))
            self.summary.setItem(i, 1, QTableWidgetItem(v))
        # plots need laid-out canvases: draw now if visible, else when first shown
        self._plot_pending = True
        if self.isVisible():
            self._draw_pending()
        f = res.failures
        self.fail_table.setColumnCount(len(f.columns))
        self.fail_table.setRowCount(len(f))
        self.fail_table.setHorizontalHeaderLabels(list(f.columns))
        for i, r in enumerate(f.itertuples(index=False)):
            for j, v in enumerate(r):
                txt = "—" if v is None or (isinstance(v, float) and math.isnan(v)) else (
                    f"{v:.0f}" if isinstance(v, float) else str(v))
                self.fail_table.setItem(i, j, QTableWidgetItem(txt))
        self.fail_table.resizeColumnsToContents()
        self.btn_csv.setEnabled(True)
        self.btn_xlsx.setEnabled(True)

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        self._draw_pending()

    def _draw_pending(self) -> None:
        if self._plot_pending and self.result is not None:
            self._plot_pending = False
            self._plot(self.result)

    def _axes(self, name, n=1):
        fig, canvas = self.figures[name]
        fig.clear()
        return fig, canvas, [fig.add_subplot(n, 1, i + 1) for i in range(n)]

    def _plot(self, res: CaseResult) -> None:
        ts = res.series
        t = ts.Time / 60.0
        rup = [x for x in (res.rupture_tresca_s, res.rupture_von_mises_s) if x is not None]

        def mark(ax):
            for x, style in zip(rup, ("-", ":")):
                ax.axvline(x / 60.0, color="k", ls=style, lw=1, alpha=0.6)
            ax.grid(alpha=0.3)
            ax.set_xlabel("time [min]")

        fig, canvas, (ax,) = self._axes("Pressure")
        ax.plot(t, ts.P_bara, color="tab:blue")
        ax.set_ylabel("pressure [bara]")
        s = res.case.psv
        if s.enabled:
            ax.axhline(s.set_bara, color="tab:red", ls="--", lw=1, label="PSV set")
            ax.legend()
        mark(ax)
        canvas.draw_idle()

        fig, canvas, (ax,) = self._axes("Temperatures")
        ax.plot(t, ts.T_gas_C, label="gas")
        if ts.T_liq_C.notna().any():
            ax.plot(t, ts.T_liq_C, label="liquid")
        shown = regions_with_area(res)
        for col, label in REGIONS:
            if col in shown:
                ax.plot(t, ts[f"{col}_T_mean_C"], ls="--", label=f"{label} (mean)")
        ax.set_ylabel("temperature [°C]")
        ax.legend(fontsize=8)
        mark(ax)
        canvas.draw_idle()

        fig, canvas, (a1, a2) = self._axes("Inventory", 2)
        a1.plot(t, ts.m_gas, label="gas")
        a1.plot(t, ts.m_liq, label="liquid")
        a1.set_ylabel("mass [kg]")
        a1.legend(fontsize=8)
        mark(a1)
        a2.plot(t, ts.level, color="tab:blue")
        a2.set_ylabel("liquid level [m]")
        mark(a2)
        canvas.draw_idle()

        fig, canvas, (a1, a2) = self._axes("Release", 2)
        a1.plot(t, ts.mdot_bdv, label="BDV")
        a1.plot(t, ts.mdot_psv, label="PSV")
        a1.set_ylabel("release rate [kg/s]")
        a1.legend(fontsize=8)
        mark(a1)
        a2.plot(t, ts.released, color="tab:gray")
        a2.set_ylabel("released [kg]")
        mark(a2)
        canvas.draw_idle()

        fig, canvas, (ax,) = self._axes("Stress")
        ax.plot(t, ts.sigma_Tresca, label="Tresca (membrane)")
        ax.plot(t, ts.sigma_vM, label="von Mises (membrane)")
        ax.plot(t, ts.sigma_allow, color="tab:red", label="allowable (hottest wall)")
        ax.set_ylabel("stress [MPa]")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        mark(ax)
        canvas.draw_idle()

    # ------------------------------------------------------------------ export
    def export_csv(self, path) -> None:
        self.result.series.to_csv(path, index=False)

    def export_xlsx(self, path) -> None:
        res = self.result
        with pd.ExcelWriter(path) as xw:
            pd.DataFrame(summary_lines(res), columns=["item", "value"]).to_excel(
                xw, sheet_name="summary", index=False)
            res.series.to_excel(xw, sheet_name="time series", index=False)
            res.failures.to_excel(xw, sheet_name="failure times", index=False)
            res.stress.to_excel(xw, sheet_name="stress", index=False)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export time series", "results.csv",
                                              "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def _export_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "results.xlsx",
                                              "Excel (*.xlsx)")
        if path:
            try:
                self.export_xlsx(path)
            except (OSError, ValueError, ImportError) as e:
                QMessageBox.warning(self, "Export", f"Could not write {path}:\n{e}")


__all__ = ["ResultsView", "regions_with_area", "summary_lines"]
