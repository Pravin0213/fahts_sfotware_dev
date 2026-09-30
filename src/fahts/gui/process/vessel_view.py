"""3-D view of the process vessel: solid steel wall (real thickness, optionally exaggerated),
wall regions before a run, wall temperature after it (through the thickness or through-wall
mean), with a cut-away to see the inside and the temperature gradient across the wall.

The VTK widget is created lazily (first time the view is shown) so the workspace can be built
without a display (offscreen widget tests never show it).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout,
                             QWidget)

from fahts.renderer.vessel_geometry import CUTAWAYS, FieldGeometry3D, VesselGeometry3D

REGION_COLOURS = ["#b8b8b8", "#5b8fd6", "#ff9a3c", "#d9412b"]  # dry, wet, peak dry, peak wet
REGION_LABELS = ["dry wall", "wetted wall", "jet zone (dry)", "jet zone (wetted)"]
COLOURINGS = ["Wall regions", "Wall temperature — through thickness",
              "Wall temperature — through-wall mean"]
THICKNESS_SCALES = [1, 3, 5, 10]


class VesselView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self._geom: VesselGeometry3D | None = None
        self._field: FieldGeometry3D | None = None  # results: the wall solver's own field
        self._case = None
        self._result = None
        self._row = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(COLOURINGS)
        self.cutaway = QComboBox()
        self.cutaway.addItems([f"cut-away: {k}" for k in CUTAWAYS])
        self.cutaway.setToolTip("Remove part of the wall to see the inside, the liquid and the "
                                "temperature through the wall on the cut faces")
        self.scale = QComboBox()
        self.scale.addItems([f"thickness ×{s}" for s in THICKNESS_SCALES])
        self.scale.setToolTip("Display exaggeration of the wall thickness (the model always "
                              "uses the real thickness)")
        self.show_liquid = QCheckBox("Liquid")
        self.show_liquid.setChecked(True)
        self.show_heads = QCheckBox("Heads")
        self.show_heads.setChecked(True)
        for w in (self.mode, self.cutaway, self.scale, self.show_liquid, self.show_heads):
            bar.addWidget(w)
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
        self.legend.setWordWrap(True)
        lay.addWidget(self.legend)

        self.mode.currentIndexChanged.connect(lambda *_: self._redraw())
        self.cutaway.currentIndexChanged.connect(lambda *_: self._redraw())
        self.scale.currentIndexChanged.connect(lambda *_: self._rebuild())
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
    @property
    def thickness_scale(self) -> float:
        return float(THICKNESS_SCALES[self.scale.currentIndex()])

    @property
    def cutaway_name(self) -> str:
        return list(CUTAWAYS)[self.cutaway.currentIndex()]

    def _make_geometry(self, case, peak_modelled=None) -> VesselGeometry3D:
        return VesselGeometry3D(case, peak_modelled=peak_modelled,
                                thickness_scale=self.thickness_scale)

    def show_case(self, case) -> None:
        """Geometry preview of a case (initial liquid level, regions)."""
        self._case, self._result, self._field = case, None, None
        self._geom = self._make_geometry(case)
        c = case.contents
        self._geom.set_level(c.hc_liquid_depth_m + c.water_depth_m)
        self.slider.setEnabled(False)
        self.time_label.setText("initial state (run the case for temperatures)")
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self._redraw(reset_camera=True)

    def show_result(self, result) -> None:
        """Wall temperatures over time from a run."""
        self._case, self._result = result.case, result
        ts = result.series
        self._geom = self._make_geometry(result.case, peak_modelled="peak_T_mean_C" in ts)
        self._make_field()
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(ts) - 1)
        self.slider.setValue(len(ts) - 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(1)
        self.mode.blockSignals(False)
        self.set_row(len(ts) - 1, reset_camera=True)

    def _rebuild(self) -> None:
        """New display thickness: rebuild the geometry, keep what is shown."""
        if self._result is not None:
            row = self._row
            self._geom = self._make_geometry(self._result.case,
                                             peak_modelled="peak_T_mean_C" in self._result.series)
            self._make_field()
            self.set_row(row)
        elif self._case is not None:
            mode = self.mode.currentIndex()
            self.show_case(self._case)
            self.mode.setCurrentIndex(mode)

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
    def _make_field(self) -> None:
        w = self._result.meta["wall3d"] if self._result is not None else None
        self._field = FieldGeometry3D(w["mesh"], self.thickness_scale) if w else None

    def _temperature_range(self, through_thickness: bool) -> tuple[float, float]:
        """Colour range over the whole run, so frames compare."""
        T = self._result.meta["wall3d"]["T"]
        if not through_thickness:
            m = self._field.mesh
            T = T.reshape(len(T), -1, m.nr) @ (m.r / m.r.sum())
        lo = float(T.min()) - 273.15
        return lo, max(float(T.max()) - 273.15, lo + 1.0)

    def _redraw(self, reset_camera: bool = False) -> None:
        self._update_legend()
        if self.plotter is None or self._geom is None:
            return
        p, g, cut = self.plotter, self._geom, self.cutaway_name
        p.clear_actors()
        mode = self.mode.currentIndex()
        show_T = self._field is not None and mode > 0
        if show_T:
            # the wall solver's temperature field, smooth around / along / through the wall
            T = self._result.meta["wall3d"]["T"][self._row]
            surf = self._field.surface(T, cut, through_wall_mean=(mode == 2))
        else:
            surf = g.wall_surface(cut)
        if show_T:
            lo, hi = self._temperature_range(mode == 1)
            p.add_mesh(surf, scalars="T_C", cmap="inferno", clim=(lo, hi), name="wall",
                       scalar_bar_args=dict(title="wall T [°C]", color="white", vertical=True,
                                            fmt="%.0f"))
        else:
            p.add_mesh(surf, scalars="region", cmap=ListedColormap(REGION_COLOURS),
                       clim=(-0.5, 3.5), show_scalar_bar=False, name="wall")
        if g.peak is not None:
            zone = g.shell.extract_cells(np.flatnonzero(g.shell.cell_data["region"] >= 2))
            edges = zone.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                               manifold_edges=False, non_manifold_edges=False)
            if edges.n_points:
                # lift the outline from the inner surface onto the (displayed) outer surface
                pts = edges.points.copy()
                pts[:, 1:] *= (g.R_out * 1.003) / g.R
                edges.points = pts
                if cut != "none":
                    lo_, hi_ = CUTAWAYS[cut]
                    th = np.degrees(np.arctan2(pts[:, 1], pts[:, 2])) % 360.0
                    edges = edges.extract_points(~((th > lo_) & (th < hi_)),
                                                 adjacent_cells=False)
                if edges.n_points:
                    p.add_mesh(edges, color="#ff3b1f", line_width=3, name="zone")
        if self.show_heads.isChecked() and cut == "none":
            for i, head in enumerate(g.heads):
                p.add_mesh(head, color="#9a9a9a", opacity=0.35, name=f"head{i}")
        body = g.liquid_body(cut) if self.show_liquid.isChecked() else None
        if body is not None and body.n_points:
            p.add_mesh(body, color="#3d7fe0", opacity=0.35, name="liquid")
        if reset_camera:
            p.view_isometric()
            p.reset_camera()
        p.render()

    def _update_legend(self) -> None:
        scale = self.thickness_scale
        note = "" if scale == 1 else f" Wall thickness drawn ×{scale:g} (display only)."
        if self.mode.currentIndex() == 0:
            self.legend.setText("   ".join(
                f'<span style="color:{c}">■</span> {lbl}'
                for c, lbl in zip(REGION_COLOURS, REGION_LABELS)) + note)
        elif self.mode.currentIndex() == 1:
            self.legend.setText("Colours: wall temperature from the 3-D wall solver: outer "
                                "surface = fire side, inner = fluid side, cut faces show the "
                                "gradient through the wall. Jet zone outlined in red." + note)
        else:
            self.legend.setText("Colours: through-wall mean temperature at each point of the "
                                "wall. Jet zone outlined in red." + note)
