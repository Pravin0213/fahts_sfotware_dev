"""Contents of the vessel: initial conditions, liquid inventory and fluid composition, with an
initial-state check that builds the model and reports what the solver will start from."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QHeaderView, QLabel,
                             QMenu, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
                             QVBoxLayout)

from fahts.coupling.vessel_case import VesselCase
from fahts.coupling.vessel_fire_1d import VesselFireModel
from fahts.gui.process.fields import dspin, form_group
from fahts.gui.process.forms import _Form
from fahts.process.geometry import VesselGeometry
from fahts.thermo.component_data import COMPONENTS

NAMES = {"H2": "hydrogen", "N2": "nitrogen", "CO2": "carbon dioxide", "H2S": "hydrogen sulphide",
         "H2O": "water", "C1": "methane", "C2": "ethane", "C3": "propane", "IC4": "i-butane",
         "C4": "n-butane", "IC5": "i-pentane", "C5": "n-pentane", "C6": "n-hexane",
         "C7": "n-heptane", "C8": "n-octane", "AR": "argon", "O2": "oxygen", "HE": "helium"}

PRESETS = {
    "Hydrogen": {"H2": 1.0},
    "Methane": {"C1": 1.0},
    "Nitrogen": {"N2": 1.0},
    "Carbon dioxide": {"CO2": 1.0},
    "Lean natural gas": {"C1": 0.90, "C2": 0.05, "C3": 0.02, "N2": 0.02, "CO2": 0.01},
    "LPG (propane / n-butane 60/40)": {"C3": 0.6, "C4": 0.4},
    "Commercial propane": {"C2": 0.02, "C3": 0.95, "IC4": 0.02, "C4": 0.01},
}

COL_NAME, COL_X, COL_SG, COL_TB = range(4)


class ContentsForm(_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        box, f = form_group("Initial conditions")
        self.P0 = dspin(0.01, 2000.0, 3, "bara")
        self.T0 = dspin(-200.0, 500.0, 2, "°C")
        self.Tshell = dspin(-200.0, 500.0, 2, "°C", tip="Initial steel temperature")
        self.hc = dspin(0.0, 20.0, 3, "m", tip="Depth of the hydrocarbon liquid layer (on top of "
                                              "any free water)")
        self.water = dspin(0.0, 20.0, 3, "m", tip="Depth of free water at the bottom")
        self.fill = QLabel()
        self.fill.setWordWrap(True)
        f.addRow("Pressure", self.P0)
        f.addRow("Fluid temperature", self.T0)
        f.addRow("Shell temperature", self.Tshell)
        f.addRow("Hydrocarbon liquid depth", self.hc)
        f.addRow("Free-water depth", self.water)
        f.addRow("", self.fill)
        lay.addWidget(box)

        box, f = form_group("Composition (mole fractions)")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Component", "Mole fr.", "SG / API", "Tb [K]"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(200)
        self.table.setToolTip("Pseudo-components (petroleum fractions): any name that is not a "
                              "library component; give SG (< 1.5) or API gravity and normal "
                              "boiling point")
        f.addRow(self.table)
        row = QGridLayout()
        self.btn_add = QPushButton("Add component")
        self.btn_pseudo = QPushButton("Add pseudo-component")
        self.btn_remove = QPushButton("Remove")
        self.btn_norm = QPushButton("Normalise")
        self.btn_preset = QPushButton("Presets ▾")
        menu = QMenu(self.btn_preset)
        for name in PRESETS:
            menu.addAction(name, lambda n=name: self._apply_preset(n))
        self.btn_preset.setMenu(menu)
        for i, b in enumerate((self.btn_add, self.btn_pseudo, self.btn_preset, self.btn_remove,
                               self.btn_norm)):
            row.addWidget(b, i // 3, i % 3)
        f.addRow(row)
        self.sum_label = QLabel()
        f.addRow(self.sum_label)
        lay.addWidget(box)

        box, f = form_group("Initial state (what the model starts from)")
        self.btn_check = QPushButton("Check initial state")
        self.state = QTextEdit()
        self.state.setReadOnly(True)
        self.state.setMinimumHeight(170)
        f.addRow(self.btn_check)
        f.addRow(self.state)
        lay.addWidget(box)
        lay.addStretch(1)

        self._case_for_check: VesselCase | None = None   # set by the workspace
        self.btn_add.clicked.connect(lambda: self._add_row("C1", 0.0))
        self.btn_pseudo.clicked.connect(lambda: self._add_row(self._new_pseudo_name(), 0.0,
                                                             0.75, 400.0))
        self.btn_remove.clicked.connect(self._remove_rows)
        self.btn_norm.clicked.connect(self._normalise)
        self.table.itemChanged.connect(lambda *_: (self._update_sum(), self.changed.emit()))
        for w in (self.P0, self.T0, self.Tshell, self.hc, self.water):
            w.valueChanged.connect(self._update_fill)
        self._watch(self.P0, self.T0, self.Tshell, self.hc, self.water)
        self._diameter = 2.0
        self._length = 6.0

    # ------------------------------------------------------------------ table helpers
    def _name_widget(self, name: str) -> QComboBox:
        w = QComboBox()
        w.setEditable(True)
        for key in COMPONENTS:
            w.addItem(f"{key}", key)
            w.setItemData(w.count() - 1, NAMES.get(key, key), Qt.ItemDataRole.ToolTipRole)
        w.setCurrentText(name)
        w.currentTextChanged.connect(lambda *_: (self._sync_row_kinds(), self.changed.emit()))
        return w

    def _add_row(self, name: str, x: float, sg: float | None = None, tb: float | None = None):
        r = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(r)
        self.table.setCellWidget(r, COL_NAME, self._name_widget(name))
        self.table.setItem(r, COL_X, QTableWidgetItem(f"{x:.8g}"))
        self.table.setItem(r, COL_SG, QTableWidgetItem("" if sg is None else f"{sg:g}"))
        self.table.setItem(r, COL_TB, QTableWidgetItem("" if tb is None else f"{tb:g}"))
        self.table.blockSignals(False)
        self._sync_row_kinds()
        self._update_sum()
        self.changed.emit()

    def _new_pseudo_name(self) -> str:
        used = {self._row(r)[0] for r in range(self.table.rowCount())}
        i = 1
        while f"PSEU{i}" in used:
            i += 1
        return f"PSEU{i}"

    def _row(self, r: int):
        name = self.table.cellWidget(r, COL_NAME).currentText().strip().upper()

        def num(c):
            it = self.table.item(r, c)
            try:
                return float(it.text().replace(",", ".")) if it and it.text().strip() else None
            except ValueError:
                return None
        return name, num(COL_X), num(COL_SG), num(COL_TB)

    def _sync_row_kinds(self):
        """Pseudo-component columns are editable only for names that are not library ones."""
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            pseudo = self._row(r)[0] not in COMPONENTS
            for c in (COL_SG, COL_TB):
                it = self.table.item(r, c)
                if it is None:
                    it = QTableWidgetItem("")
                    self.table.setItem(r, c, it)
                flags = it.flags()
                it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if pseudo
                            else flags & ~Qt.ItemFlag.ItemIsEditable)
                if not pseudo:
                    it.setText("")
        self.table.blockSignals(False)

    def _remove_rows(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._update_sum()
        self.changed.emit()

    def _normalise(self):
        rows = [self._row(r) for r in range(self.table.rowCount())]
        total = sum(x or 0.0 for _, x, _, _ in rows)
        if total <= 0:
            return
        self.table.blockSignals(True)
        for r, (_, x, _, _) in enumerate(rows):
            self.table.item(r, COL_X).setText(f"{(x or 0.0) / total:.8f}")
        self.table.blockSignals(False)
        self._update_sum()
        self.changed.emit()

    def _apply_preset(self, name: str):
        self.table.setRowCount(0)
        for k, x in PRESETS[name].items():
            self._add_row(k, x)

    def _update_sum(self):
        total = sum(self._row(r)[1] or 0.0 for r in range(self.table.rowCount()))
        ok = abs(total - 1.0) <= 1e-6
        self.sum_label.setText(f"Sum of mole fractions: {total:.8f}" + ("" if ok else
                                                                          "  ← use Normalise"))

    def _update_fill(self):
        D, L = self._diameter, self._length
        g = VesselGeometry(D=D, L=L, orientation="horizontal", head="flat")
        h = self.hc.value() + self.water.value()
        if h >= D:
            self.fill.setText("liquid fills the vessel: no gas space")
            return
        v_liq = g.volume(h) if h > 0 else 0.0
        self.fill.setText(f"liquid {v_liq:.2f} m³ of {g.V_total:.2f} m³ "
                          f"({100 * v_liq / g.V_total:.1f} % full, level {h:.3f} m)")

    # ------------------------------------------------------------------ case <-> widgets
    def set_geometry(self, D: float, L: float):
        self._diameter, self._length = D, L
        self._update_fill()

    def set_case(self, case):
        c = case.contents
        self.P0.setValue(c.P0_bara)
        self.T0.setValue(c.T0_C)
        self.Tshell.setValue(c.T_shell_C)
        self.hc.setValue(c.hc_liquid_depth_m)
        self.water.setValue(c.water_depth_m)
        self.table.setRowCount(0)
        for name, x in c.composition.items():
            p = c.pseudo.get(name)
            self._add_row(name, x, p["sg"] if p else None, p["Tb_K"] if p else None)
        self.set_geometry(case.vessel.inner_diameter_m, case.vessel.length_m)
        self.state.clear()

    def apply(self, case):
        c = case.contents
        c.P0_bara = self.P0.value()
        c.T0_C = self.T0.value()
        c.T_shell_C = self.Tshell.value()
        c.hc_liquid_depth_m = self.hc.value()
        c.water_depth_m = self.water.value()
        comp, pseudo = {}, {}
        for r in range(self.table.rowCount()):
            name, x, sg, tb = self._row(r)
            if not name:
                continue
            comp[name] = x if x is not None else 0.0  # blank counts as 0 (sum check)
            if name not in COMPONENTS:
                pseudo[name] = {"sg": sg or 0.0, "Tb_K": tb or 0.0}
        c.composition, c.pseudo = comp, pseudo

    # ------------------------------------------------------------------ initial state
    def check_initial_state(self, case: VesselCase) -> str:
        """Build the model from ``case`` and describe its initial state (also shown)."""
        errors = [e for e in case.validate() if "heat load" not in e]
        if errors:
            text = "Fix the inputs first:\n  " + "\n  ".join(errors)
        else:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                m = VesselFireModel(case.to_model_case(), case.model_options(),
                                    case.vessel.material.steel_table())
                text = _describe(m)
            except Exception as e:  # noqa: BLE001 - show any model error to the user
                text = f"The model cannot start from these inputs:\n  {type(e).__name__}: {e}"
            finally:
                QGuiApplication.restoreOverrideCursor()
        self.state.setPlainText(text)
        return text


def _describe(m: VesselFireModel) -> str:
    G, Lz = m.G, m.Lz
    lines = [f"Vessel volume {m.V:.3f} m³, pressure {m.P / 1e5:.3f} bara"]
    for label, z in (("Gas zone", G), ("Liquid zone", Lz)):
        if z.empty:
            lines.append(f"{label}: empty")
            continue
        phases = ", ".join(f"{p.name} {100 * p.beta:.1f} mol-%" for p in z.r.phases)
        lines.append(f"{label}: {z.mass:,.1f} kg, {z.V:.3f} m³, T {z.T - 273.15:.2f} °C, "
                     f"density {z.mass / z.V:.2f} kg/m³  [{phases}]")
    lines.append(f"Liquid level {m._level():.3f} m, wetted wall {100 * m.frac['wet']:.1f} %")
    if m.peak is not None:
        lines.append(f"Peak fire zone: {100 * (m.frac['peak_dry'] + m.frac['peak_wet']):.2f} % "
                     f"of the shell ({100 * m.frac['peak_wet']:.2f} % wetted)")
    lines.append(f"Wall: {len(m.x_nodes)} nodes through {m.t_w * 1e3:.1f} mm, "
                 f"steel {m.mat.name}")
    return "\n".join(lines)
