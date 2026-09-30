"""3-D view of the process vessel: wall regions before a run, wall temperature after it.

The VTK widget is created lazily (first time the view is shown) so the workspace can be built
without a display (offscreen widget tests never show it).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from fahts.coupling.report import regions_with_area
from fahts.renderer.vessel_geometry import (REGION_COLUMNS, REGION_KEYS, VesselGeometry3D,
                                            region_temperatures)

REGION_COLOURS = ["#b8b8b8", "#5b8fd6", "#ff9a3c", "#d9412b"]  # dry, wet, peak dry, peak wet
REGION_LABELS = ["dry wall", "wetted wall", "jet zone (dry)", "jet zone (wetted)"]
SURFACES = {"through-wall mean": "mean", "outer surface": "out", "inner surface": "in"}


class VesselView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self._geom: VesselGeometry3D | None = None
        self._case = None
        self._result = None
        self._row = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Wall regions"] + [f"Wall temperature — {s}" for s in SURFACES])
        self.show_liquid = QCheckBox("Liquid")
        self.show_liquid.setChecked(True)
        self.show_heads = QCheckBox("Heads")
        self.show_heads.setChecked(True)
        bar.addWidget(self.mode)
        bar.addWidget(self.show_liquid)
        bar.addWidget(self.show_heads)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.view_area = QVBoxLayout()
        lay.addLayout(self.view_area, 1)
        tbar = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(380)
        tbar.addWidget(QLabel("Time"))
        tbar.addWidget(self.slider, 1)
        tbar.addWidget(self.time_label)
        lay.addLayout(tbar)
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.legend)

        self.mode.currentIndexChanged.connect(lambda *_: self._redraw())
        self.show_liquid.toggled.connect(lambda *_: self._redraw())
        self.show_heads.toggled.connect(lambda *_: self._redraw())
        self.slider.valueChanged.connect(self.set_row)

    # ------------------------------------------------------------------ lazy VTK widget
    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        self.ensure_plotter()

    def ensure_plotter(self) -> bool:
        if self.plotter is None:
            from pyvistaqt import QtInteractor   # imported here: needs a display
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#2b2b2b")
            self.plotter.add_axes()
            self.view_area.addWidget(self.plotter.interactor)
            self._redraw(reset_camera=True)
        return True

    # ------------------------------------------------------------------ content
    def show_case(self, case) -> None:
        """Geometry preview of a case (initial liquid level, regions)."""
        self._case, self._result = case, None
        self._geom = VesselGeometry3D(case)
        c = case.contents
        self._geom.set_level(c.hc_liquid_depth_m + c.water_depth_m)
        self.slider.setEnabled(False)
        self.time_label.setText("initial state (run the case for temperatures)")
        self.mode.setCurrentIndex(0)
        self._redraw(reset_camera=True)

    def show_result(self, result) -> None:
        """Wall temperatures over time from a run."""
        self._case, self._result = result.case, result
        ts = result.series
        self._geom = VesselGeometry3D(result.case, peak_modelled="peak_T_mean_C" in ts)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(ts) - 1)
        self.slider.setValue(len(ts) - 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(1)
        self.mode.blockSignals(False)
        self.set_row(len(ts) - 1, reset_camera=True)

    def set_row(self, row: int, reset_camera: bool = False) -> None:
        if self._result is None or self._geom is None:
            return
        ts = self._result.series
        self._row = int(row)
        self._geom.set_level(float(ts.level.iloc[self._row]))
        self.time_label.setText(
            f"t = {ts.Time.iloc[self._row]:.0f} s   P = {ts.P_bara.iloc[self._row]:.2f} bara   "
            f"level = {ts.level.iloc[self._row]:.3f} m")
        self._redraw(reset_camera=reset_camera)

    # ------------------------------------------------------------------ drawing
    def _temperature_range(self, which: str) -> tuple[float, float]:
        ts = self._result.series
        cols = [f"{REGION_COLUMNS[k]}_T_{which}_C" for k in REGION_KEYS]
        keep = {REGION_COLUMNS[k] for k in REGION_KEYS} & set(regions_with_area(self._result))
        vals = np.concatenate([ts[c].to_numpy() for c in cols
                               if c in ts and c.split("_T_")[0] in keep])
        return float(np.nanmin(vals)), float(max(np.nanmax(vals), np.nanmin(vals) + 1.0))

    def _redraw(self, reset_camera: bool = False) -> None:
        self._update_legend()
        if self.plotter is None or self._geom is None:
            return
        p, g = self.plotter, self._geom
        p.clear_actors()
        mode = self.mode.currentIndex()
        if mode > 0 and self._result is not None:
            which = list(SURFACES.values())[mode - 1]
            g.paint(region_temperatures(self._result.series, self._row, which))
            lo, hi = self._temperature_range(which)
            p.add_mesh(g.shell, scalars="T_C", cmap="inferno", clim=(lo, hi), name="shell",
                       scalar_bar_args=dict(title="wall T [°C]", color="white", vertical=True))
        else:
            p.add_mesh(g.shell, scalars="region", cmap=ListedColormap(REGION_COLOURS),
                       clim=(-0.5, 3.5), show_scalar_bar=False, name="shell")
        if g.peak is not None:
            zone = g.shell.extract_cells(np.flatnonzero(g.shell.cell_data["region"] >= 2))
            edges = zone.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                               manifold_edges=False, non_manifold_edges=False)
            if edges.n_points:
                p.add_mesh(edges, color="#ff3b1f", line_width=3, name="zone")
        if self.show_heads.isChecked():
            for i, head in enumerate(g.heads):
                p.add_mesh(head, color="#9a9a9a", opacity=0.35, name=f"head{i}")
        body = g.liquid_body() if self.show_liquid.isChecked() else None
        if body is not None:
            p.add_mesh(body, color="#3d7fe0", opacity=0.35, name="liquid")
        if reset_camera:
            p.view_isometric()
            p.reset_camera()
        p.render()

    def _update_legend(self) -> None:
        if self.mode.currentIndex() == 0:
            self.legend.setText("   ".join(
                f'<span style="color:{c}">■</span> {lbl}'
                for c, lbl in zip(REGION_COLOURS, REGION_LABELS)))
        else:
            self.legend.setText("Colours: wall temperature of the model's wall regions "
                                "(dry / wetted, background / jet zone); jet zone outlined in red.")
