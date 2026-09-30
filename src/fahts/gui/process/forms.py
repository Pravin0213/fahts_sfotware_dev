"""Input forms of the process workspace. Each form edits one part of a ``VesselCase``:

    form.set_case(case)   # case -> widgets
    form.apply(case)      # widgets -> case (in place)

Units in the widgets are engineering units (mm for wall / orifices / line), converted here.
"""

from __future__ import annotations

from dataclasses import fields

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QDoubleSpinBox, QFileDialog, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QRadioButton, QVBoxLayout, QWidget)

from fahts.coupling.options import VesselFireOptions
from fahts.coupling.vessel_case import (PSV_TYPES, BlowdownLineSpec, MaterialSpec, VesselCase)
from fahts.gui.process.fields import combo, dspin, form_group
from fahts.materials import SteelTable


class _Form(QWidget):
    changed = pyqtSignal()

    def set_case(self, case: VesselCase) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def apply(self, case: VesselCase) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _watch(self, *widgets) -> None:
        """Emit ``changed`` when any of the widgets is edited."""
        for w in widgets:
            for sig in ("valueChanged", "currentIndexChanged", "toggled", "textEdited"):
                if hasattr(w, sig):
                    getattr(w, sig).connect(lambda *_: self.changed.emit())
                    break


# ====================================================================== vessel + material
class VesselForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        box, f = form_group("Vessel (horizontal, flat heads)")
        self.tag = QLineEdit()
        self.D = dspin(0.05, 20.0, 3, "m", tip="Inner diameter")
        self.wall = dspin(0.5, 500.0, 2, "mm", tip="Steel wall thickness")
        self.L = dspin(0.1, 200.0, 3, "m", tip="Shell (tangent-tangent) length")
        self.strength = dspin(1.0, 2000.0, 1, "MPa",
                              tip="Reference strength for the allowable stress (UTS or yield "
                                  "at room temperature, see Stress)")
        f.addRow("Tag", self.tag)
        f.addRow("Inner diameter", self.D)
        f.addRow("Wall thickness", self.wall)
        f.addRow("Shell length", self.L)
        f.addRow("Reference strength", self.strength)
        lay.addWidget(box)

        mbox, mf = form_group("Steel properties vs temperature")
        self.mat_label = QLabel()
        self.mat_label.setWordWrap(True)
        self.btn_builtin = QPushButton("Use built-in EN 1993-1-2 carbon steel")
        self.btn_csv = QPushButton("Load grade table (CSV)…")
        self.btn_export = QPushButton("Export table (CSV)…")
        mf.addRow(self.mat_label)
        for b in (self.btn_builtin, self.btn_csv, self.btn_export):
            mf.addRow(b)
        lay.addWidget(mbox)
        lay.addStretch(1)
        self._material = MaterialSpec()
        self.btn_builtin.clicked.connect(self._use_builtin)
        self.btn_csv.clicked.connect(self._load_csv)
        self.btn_export.clicked.connect(self._export_csv)
        self._watch(self.tag, self.D, self.wall, self.L, self.strength)

    def _show_material(self):
        t = self._material.steel_table()
        self.mat_label.setText(
            f"<b>{t.name}</b><br>{len(t.T)} points, {t.T[0] - 273.15:.0f}-{t.T[-1] - 273.15:.0f} °C, "
            f"ρ = {t.rho:g} kg/m³")

    def _use_builtin(self):
        self._material = MaterialSpec()
        self._show_material()
        self.changed.emit()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Steel property table", "", "CSV (*.csv)")
        if not path:
            return
        try:
            table = SteelTable.from_csv(path)
        except (ValueError, OSError, KeyError) as e:
            QMessageBox.warning(self, "Steel table", f"Could not read {path}:\n{e}")
            return
        self._material = MaterialSpec(kind="table", table=table.to_dict())
        self._show_material()
        self.changed.emit()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export steel table", "steel.csv", "CSV (*.csv)")
        if path:
            self._material.steel_table().to_csv(path)

    def set_case(self, case):
        v = case.vessel
        self.tag.setText(v.tag)
        self.D.setValue(v.inner_diameter_m)
        self.wall.setValue(v.wall_m * 1e3)
        self.L.setValue(v.length_m)
        self.strength.setValue(v.strength_mpa)
        self._material = v.material
        self._show_material()

    def apply(self, case):
        v = case.vessel
        v.tag = self.tag.text()
        v.inner_diameter_m = self.D.value()
        v.wall_m = self.wall.value() / 1e3
        v.length_m = self.L.value()
        v.strength_mpa = self.strength.value()
        v.material = self._material


# ====================================================================== ambient
class AmbientForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        box, f = form_group("Surroundings (outside the vessel)")
        self.T = dspin(-60.0, 80.0, 1, "°C")
        self.rb_wind = QRadioButton("Wind speed (forced + natural convection)")
        self.rb_fixed = QRadioButton("Fixed convection coefficient")
        self.wind = dspin(0.0, 50.0, 2, "m/s")
        self.h = dspin(0.1, 500.0, 1, "W/m²K")
        self.eps = dspin(0.0, 1.0, 2, tip="Outer surface emissivity (exchange with surroundings)")
        f.addRow("Air temperature", self.T)
        f.addRow(self.rb_wind)
        f.addRow("  Wind speed", self.wind)
        f.addRow(self.rb_fixed)
        f.addRow("  Coefficient", self.h)
        f.addRow("Surface emissivity", self.eps)
        lay.addWidget(box)
        lay.addStretch(1)
        self.rb_wind.toggled.connect(self._sync)
        self._watch(self.T, self.rb_wind, self.wind, self.h, self.eps)

    def _sync(self):
        self.wind.setEnabled(self.rb_wind.isChecked())
        self.h.setEnabled(not self.rb_wind.isChecked())

    def set_case(self, case):
        a = case.ambient
        self.T.setValue(a.T_amb_C)
        (self.rb_wind if a.convection == "wind" else self.rb_fixed).setChecked(True)
        self.wind.setValue(a.wind_m_s)
        self.h.setValue(a.h_W_m2K)
        self.eps.setValue(a.emissivity)
        self._sync()

    def apply(self, case):
        a = case.ambient
        a.T_amb_C = self.T.value()
        a.convection = "wind" if self.rb_wind.isChecked() else "fixed"
        a.wind_m_s = self.wind.value()
        a.h_W_m2K = self.h.value()
        a.emissivity = self.eps.value()


# ====================================================================== relief
class ReliefForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        box, f = form_group("Discharge")
        self.pb = dspin(0.5, 50.0, 4, "bara", tip="Back pressure at the valve / line exit")
        f.addRow("Back pressure", self.pb)
        lay.addWidget(box)

        self.bdv_box, f = form_group("Blowdown valve (BDV)", checkable=True)
        self.bdv_d = dspin(0.1, 500.0, 2, "mm", tip="Orifice diameter")
        self.bdv_cd = dspin(0.01, 1.0, 3, tip="Discharge (contraction) coefficient")
        self.bdv_delay = dspin(0.0, 1e5, 1, "s", tip="Opening time (instant opening)")
        f.addRow("Orifice diameter", self.bdv_d)
        f.addRow("Discharge coefficient", self.bdv_cd)
        f.addRow("Opens at", self.bdv_delay)
        self.line_box, lf = form_group("Blowdown line (friction)", checkable=True)
        self.line_od = dspin(1.0, 2000.0, 2, "mm", tip="Line OUTER diameter (bore = OD - 2 wall)")
        self.line_t = dspin(0.0, 100.0, 2, "mm")
        self.line_L = dspin(0.01, 1e4, 2, "m")
        lf.addRow("Outer diameter", self.line_od)
        lf.addRow("Wall thickness", self.line_t)
        lf.addRow("Length", self.line_L)
        f.addRow(self.line_box)
        lay.addWidget(self.bdv_box)

        self.psv_box, f = form_group("Pressure safety valve (PSV)", checkable=True)
        self.psv_d = dspin(0.1, 500.0, 2, "mm")
        self.psv_cd = dspin(0.01, 1.0, 3)
        self.psv_set = dspin(0.5, 2000.0, 3, "bara")
        self.psv_full = dspin(0.5, 2000.0, 3, "bara")
        self.psv_reseat = dspin(0.5, 2000.0, 3, "bara")
        self.psv_type = combo(list(PSV_TYPES),
                              tip="Opening characteristic: trapezoidal = opens linearly from set "
                                  "to full-open; triangular / square = opens fully at set")
        f.addRow("Orifice diameter", self.psv_d)
        f.addRow("Discharge coefficient", self.psv_cd)
        f.addRow("Set pressure", self.psv_set)
        f.addRow("Full-open pressure", self.psv_full)
        f.addRow("Reseat pressure", self.psv_reseat)
        f.addRow("Characteristic", self.psv_type)
        lay.addWidget(self.psv_box)
        lay.addStretch(1)
        self._watch(self.pb, self.bdv_box, self.bdv_d, self.bdv_cd, self.bdv_delay, self.line_box,
                    self.line_od, self.line_t, self.line_L, self.psv_box, self.psv_d, self.psv_cd,
                    self.psv_set, self.psv_full, self.psv_reseat, self.psv_type)

    def set_case(self, case):
        self.pb.setValue(case.back_pressure_bara)
        b = case.blowdown
        self.bdv_box.setChecked(b.enabled)
        self.bdv_d.setValue(b.orifice_mm)
        self.bdv_cd.setValue(b.cd)
        self.bdv_delay.setValue(b.delay_s)
        ln = b.line or BlowdownLineSpec()
        self.line_box.setChecked(b.line is not None)
        self.line_od.setValue(ln.outer_diameter_mm)
        self.line_t.setValue(ln.wall_mm)
        self.line_L.setValue(ln.length_m)
        p = case.psv
        self.psv_box.setChecked(p.enabled)
        self.psv_d.setValue(p.orifice_mm)
        self.psv_cd.setValue(p.cd)
        self.psv_set.setValue(p.set_bara)
        self.psv_full.setValue(p.full_open_bara)
        self.psv_reseat.setValue(p.reseat_bara)
        self.psv_type.setCurrentText(p.characteristic)

    def apply(self, case):
        case.back_pressure_bara = self.pb.value()
        b = case.blowdown
        b.enabled = self.bdv_box.isChecked()
        b.orifice_mm = self.bdv_d.value()
        b.cd = self.bdv_cd.value()
        b.delay_s = self.bdv_delay.value()
        b.line = BlowdownLineSpec(self.line_od.value(), self.line_t.value(),
                                  self.line_L.value()) if self.line_box.isChecked() else None
        p = case.psv
        p.enabled = self.psv_box.isChecked()
        p.orifice_mm = self.psv_d.value()
        p.cd = self.psv_cd.value()
        p.set_bara = self.psv_set.value()
        p.full_open_bara = self.psv_full.value()
        p.reseat_bara = self.psv_reseat.value()
        p.characteristic = self.psv_type.currentText()


# ====================================================================== stress + run
class StressRunForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        box, f = form_group("Rupture check")
        self.basis = combo(["UTS", "yield"], tip="Allowable = reference strength x factor x "
                                                  "retention(T) of UTS or yield strength")
        self.factor = dspin(0.01, 10.0, 3)
        self.ext = dspin(-1000.0, 1000.0, 1, "MPa", tip="External longitudinal membrane stress")
        f.addRow("Strength basis", self.basis)
        f.addRow("Strength factor", self.factor)
        f.addRow("External longitudinal stress", self.ext)
        lay.addWidget(box)
        box, f = form_group("Simulation")
        self.t_end = dspin(1.0, 1e6, 0, "s")
        self.dt = dspin(0.01, 60.0, 2, "s")
        self.out = dspin(0.1, 3600.0, 1, "s", tip="Output interval")
        f.addRow("End time", self.t_end)
        f.addRow("Time step", self.dt)
        f.addRow("Output interval", self.out)
        lay.addWidget(box)
        lay.addStretch(1)
        self._watch(self.basis, self.factor, self.ext, self.t_end, self.dt, self.out)

    def set_case(self, case):
        self.basis.setCurrentText(case.stress.basis)
        self.factor.setValue(case.stress.factor)
        self.ext.setValue(case.stress.ext_long_mpa)
        self.t_end.setValue(case.run.t_end_s)
        self.dt.setValue(case.run.dt_s)
        self.out.setValue(case.run.output_s)

    def apply(self, case):
        case.stress.basis = self.basis.currentText()
        case.stress.factor = self.factor.value()
        case.stress.ext_long_mpa = self.ext.value()
        case.run.t_end_s = self.t_end.value()
        case.run.dt_s = self.dt.value()
        case.run.output_s = self.out.value()


# ====================================================================== model options
# (option, choices or None for numbers, help). Defaults come from VesselFireOptions; only
# values that differ from the default are stored in the case.
MODEL_OPTIONS = [
    ("wall_model", ["1d", "3d"],
     "Steel wall: 1d = radial conduction per wall region (fast, validated against VessFire); "
     "3d = Hex8 solid shell, heat also flows around and along the wall (realistic hot spots)"),
    ("wall3d_n_theta", None, "3-D wall: divisions around the circumference"),
    ("wall3d_n_length", None, "3-D wall: divisions along the shell"),
    ("wall3d_n_radial", None, "3-D wall: layers through the thickness"),
    ("h_corr", ["evans_stefany", "churchill_chu", "mcadams", "laminar", "woodfield_h2"],
     "Wall -> gas natural convection correlation"),
    ("wet_boiling", ["nucleate_only", "full"], "Wetted-wall boiling curve"),
    ("wet_above_crit", ["boiling", "single-phase", "supercritical"],
     "Wetted-wall heat transfer when the pool cannot boil (above its cricondenbar)"),
    ("psv_liquid", ["liquid", "gas"], "PSV flow equation when drawing from the dense zone"),
    ("water_mode", ["vf", "physical", "sink"], "Free-water pool heat"),
    ("flux", ["blackbody", "balance"], "Fire boundary: how the specified flux is applied"),
    ("line_diameter", ["outer", "inner"], "Blowdown-line diameter input is the OD or the bore"),
    ("rad_internal", [True, False], "Radiation inside the vapour space"),
    ("peak_zone", [True, False], "Model the local peak (jet) fire zone"),
    ("interface_mass", [False, True], "Interface evaporation / condensation"),
    ("eps_surf_fire", None, "Steel surface emissivity towards the fire"),
    ("h_fire", None, "Fire-side convection coefficient [W/m²K]"),
    ("wall_cells", None, "Radial cells through the steel wall"),
]


class OptionsForm(_Form):
    """Model choices (see docs/process_model_choices.md). Defaults are the validated profile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        note = QLabel("Defaults are the validated model profile (docs/process_model_choices.md). "
                      "Change only for sensitivity studies.")
        note.setWordWrap(True)
        lay.addWidget(note)
        box, f = form_group("Model options")
        self._defaults = {fl.name: fl.default for fl in fields(VesselFireOptions)}
        self.widgets = {}
        for name, choices, tip in MODEL_OPTIONS:
            d = self._defaults[name]
            if choices is None:
                w = dspin(0.0, 1e4, 0 if isinstance(d, int) else 3, tip=tip)
            else:
                w = combo([f"{c}" + ("  (default)" if c == d else "") for c in choices], tip)
                w.setProperty("choices", choices)
            self.widgets[name] = w
            f.addRow(name, w)
        lay.addWidget(box)
        self.btn_reset = QPushButton("Reset to defaults")
        self.btn_reset.clicked.connect(lambda: (self._show({}), self.changed.emit()))
        lay.addWidget(self.btn_reset)
        lay.addStretch(1)
        self._watch(*self.widgets.values())

    def _show(self, overrides: dict):
        for name, w in self.widgets.items():
            v = overrides.get(name, self._defaults[name])
            if isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            else:
                w.setCurrentIndex(w.property("choices").index(v))

    def set_case(self, case):
        self._show(case.options)

    def apply(self, case):
        opts = {k: v for k, v in case.options.items() if k not in self.widgets}
        for name, w in self.widgets.items():
            d = self._defaults[name]
            if w.property("choices") is not None:
                v = w.property("choices")[w.currentIndex()]
            else:
                v = int(w.value()) if isinstance(d, int) else w.value()
            if v != d:
                opts[name] = v
        case.options = opts


class CaseInfoForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        box, f = form_group("Case")
        self.name = QLineEdit()
        self.desc = QLineEdit()
        f.addRow("Name", self.name)
        f.addRow("Description", self.desc)
        lay.addWidget(box)
        self._watch(self.name, self.desc)

    def set_case(self, case):
        self.name.setText(case.name)
        self.desc.setText(case.description)

    def apply(self, case):
        case.name = self.name.text()
        case.description = self.desc.text()


__all__ = ["AmbientForm", "CaseInfoForm", "OptionsForm", "ReliefForm", "StressRunForm",
           "VesselForm"]
