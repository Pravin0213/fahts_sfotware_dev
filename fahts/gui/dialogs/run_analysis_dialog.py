"""
Task 3.7 — RunAnalysisDialog

Modal dialog that lets the user configure a heat-transfer analysis run and
returns an AnalysisConfig on acceptance.

Signal
------
config_accepted(AnalysisConfig)   emitted when the user clicks OK with valid params.

Usage
-----
    dlg = RunAnalysisDialog(n_exposed=42, parent=self)
    dlg.config_accepted.connect(self._on_run_analysis)
    dlg.exec()
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fahts.core.results.analysis_config import AnalysisConfig


class RunAnalysisDialog(QDialog):
    """
    Modal dialog for configuring a transient heat-transfer analysis run.

    Parameters
    ----------
    n_exposed:
        Number of beam elements currently exposed to fire.  Displayed as
        informational text; 0 means the dialog will warn on accept.
    parent:
        Qt parent widget.
    """

    config_accepted: pyqtSignal = pyqtSignal(object)  # emits AnalysisConfig

    # Default parameter values
    _DEFAULT_T_END_MIN:  float = 120.0
    _DEFAULT_DT_S:       float = 30.0
    _DEFAULT_OUT_DT_S:   float = 60.0
    # BOX surface mesh defaults (FAHTS standard)
    _DEFAULT_N_TOP:    int = 2
    _DEFAULT_N_SIDE:   int = 3
    _DEFAULT_N_LENGTH: int = 4
    # I-profile surface mesh defaults
    _DEFAULT_N_TOP_I:    int = 4
    _DEFAULT_N_SIDE_I:   int = 2
    _DEFAULT_N_BOTTOM_I: int = 2
    _DEFAULT_N_LENGTH_I: int = 2
    # PIPE surface mesh defaults
    _DEFAULT_C_CIRC:    int = 8
    _DEFAULT_N_LENGTH_P: int = 4
    # Shell / plate surface mesh defaults
    _DEFAULT_MESH_12: int = 4
    _DEFAULT_MESH_14: int = 2
    # Material thermal property defaults (USFOS fahts.fem values)
    _DEFAULT_EPSILON_STEEL:    float = 0.85
    _DEFAULT_DENSITY:          float = 7850.0
    _DEFAULT_C_REF:            float = 510.0
    _DEFAULT_K_REF:            float = 50.0
    _DEFAULT_ENCLOSED_GAS_RHO_C: float = 1200.0
    # USFOS tempdepy factor tables (from fahts.fem)
    _DEFAULT_CP_TABLE: list[tuple[float, float]] = [
        (0, 0.792), (100, 0.943), (200, 1.018), (300, 1.094), (400, 1.131),
        (500, 1.282), (600, 1.508), (650, 1.584), (685, 1.697), (731, 9.804),
        (750, 2.790), (773, 1.998), (807, 1.471), (870, 1.282), (1300, 1.282),
    ]
    _DEFAULT_K_TABLE: list[tuple[float, float]] = [
        (0, 1.084), (100, 1.019), (200, 0.949), (300, 0.874), (400, 0.809),
        (500, 0.744), (600, 0.679), (700, 0.614), (800, 0.548), (900, 0.548),
        (1000, 0.548), (1100, 0.548), (1200, 0.548), (1300, 0.548),
    ]

    def __init__(
        self,
        n_exposed: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._n_exposed = n_exposed
        self.setWindowTitle("Run Heat Transfer Analysis")
        self.setMinimumWidth(760)
        self._build_ui()
        self._on_dt_changed()   # initialise output_dt min bound

    # ── Public API ────────────────────────────────────────────────────────────

    def get_config(self) -> AnalysisConfig:
        """Build and return an AnalysisConfig from the current field values."""
        t_end     = self._t_end_min.value() * 60.0
        dt        = self._dt_s.value()
        output_dt = self._out_dt_s.value()

        if self._box_default_rb.isChecked():
            n_top    = self._DEFAULT_N_TOP
            n_side   = self._DEFAULT_N_SIDE
            n_length = self._DEFAULT_N_LENGTH
        else:
            n_top    = self._n_top_sb.value()
            n_side   = self._n_side_sb.value()
            n_length = self._n_length_sb.value()

        if self._iprofil_default_rb.isChecked():
            n_top_i    = self._DEFAULT_N_TOP_I
            n_side_i   = self._DEFAULT_N_SIDE_I
            n_bottom_i = self._DEFAULT_N_BOTTOM_I
            n_length_i = self._DEFAULT_N_LENGTH_I
        else:
            n_top_i    = self._n_top_i_sb.value()
            n_side_i   = self._n_side_i_sb.value()
            n_bottom_i = self._n_bottom_i_sb.value()
            n_length_i = self._n_length_i_sb.value()

        if self._pipe_default_rb.isChecked():
            c_circ    = self._DEFAULT_C_CIRC
            n_length_p = self._DEFAULT_N_LENGTH_P
        else:
            c_circ    = self._c_circ_sb.value()
            n_length_p = self._n_length_p_sb.value()

        if self._shell_default_rb.isChecked():
            mesh_12 = self._DEFAULT_MESH_12
            mesh_14 = self._DEFAULT_MESH_14
        else:
            mesh_12 = self._mesh_12_sb.value()
            mesh_14 = self._mesh_14_sb.value()

        cp_table = self._read_factor_table(self._cp_table)
        k_table  = self._read_factor_table(self._k_table)
        property_model = "usfos" if (cp_table or k_table) else "en1993"

        return AnalysisConfig(
            t_end=t_end,
            dt=dt,
            output_dt=output_dt,
            n_top=n_top,
            n_side=n_side,
            n_length=n_length,
            n_top_i=n_top_i,
            n_side_i=n_side_i,
            n_bottom_i=n_bottom_i,
            n_length_i=n_length_i,
            c_circ=c_circ,
            n_length_p=n_length_p,
            mesh_12=mesh_12,
            mesh_14=mesh_14,
            property_model=property_model,
            epsilon_steel=self._epsilon_sb.value(),
            density=self._density_sb.value(),
            c_ref=self._c_ref_sb.value(),
            k_ref=self._k_ref_sb.value(),
            enclosed_gas_rho_c=self._enclosed_gas_sb.value(),
            cp_factor_table=cp_table,
            k_factor_table=k_table,
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Exposure summary ─────────────────────────────────────────────────
        info_text = (
            f"<b>{self._n_exposed}</b> beam element(s) exposed to fire will be analysed."
            if self._n_exposed > 0
            else "<b style='color:red;'>No exposed elements.</b>  "
                 "Add fire zones before running the analysis."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        root.addWidget(info_label)

        # ── Time parameters ──────────────────────────────────────────────────
        time_grp = QGroupBox("Time Integration")
        time_form = QFormLayout(time_grp)
        time_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._t_end_min = QDoubleSpinBox()
        self._t_end_min.setRange(1.0, 240.0)
        self._t_end_min.setDecimals(0)
        self._t_end_min.setSuffix(" min")
        self._t_end_min.setValue(self._DEFAULT_T_END_MIN)
        self._t_end_min.setToolTip("Total fire duration (1–240 minutes)")
        time_form.addRow("Fire duration:", self._t_end_min)

        self._dt_s = QDoubleSpinBox()
        self._dt_s.setRange(1.0, 300.0)
        self._dt_s.setDecimals(0)
        self._dt_s.setSuffix(" s")
        self._dt_s.setValue(self._DEFAULT_DT_S)
        self._dt_s.setToolTip(
            "Solver time step [s].  Smaller → more accurate but slower.\n"
            "Crank-Nicolson is unconditionally stable — 30 s is typical."
        )
        self._dt_s.valueChanged.connect(self._on_dt_changed)
        time_form.addRow("Time step (dt):", self._dt_s)

        self._out_dt_s = QDoubleSpinBox()
        self._out_dt_s.setRange(1.0, 14400.0)
        self._out_dt_s.setDecimals(0)
        self._out_dt_s.setSuffix(" s")
        self._out_dt_s.setValue(self._DEFAULT_OUT_DT_S)
        self._out_dt_s.setToolTip(
            "Store results every N seconds.  Must be >= dt.\n"
            "60 s (every minute) is a good default for post-processing."
        )
        time_form.addRow("Output every:", self._out_dt_s)

        root.addWidget(time_grp)

        # ── Mesh — four profile columns side by side ──────────────────────────
        mesh_grp = QGroupBox("Mesh")
        mesh_hlayout = QHBoxLayout(mesh_grp)
        mesh_hlayout.setSpacing(8)

        # ── BOX column ───────────────────────────────────────────────────────
        box_col = QGroupBox("BOX")
        box_vbox = QVBoxLayout(box_col)
        box_vbox.setSpacing(6)

        box_rb_row = QHBoxLayout()
        self._box_default_rb = QRadioButton("Default")
        self._box_default_rb.setChecked(True)
        self._box_default_rb.setToolTip(
            f"n_top={self._DEFAULT_N_TOP}, n_side={self._DEFAULT_N_SIDE}, "
            f"n_length={self._DEFAULT_N_LENGTH}"
        )
        self._box_custom_rb = QRadioButton("Custom")
        self._box_default_rb.toggled.connect(self._on_box_mesh_mode_changed)
        box_rb_row.addWidget(self._box_default_rb)
        box_rb_row.addWidget(self._box_custom_rb)
        box_vbox.addLayout(box_rb_row)

        box_form = QFormLayout()
        box_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_top_sb = QSpinBox()
        self._n_top_sb.setRange(1, 20)
        self._n_top_sb.setValue(self._DEFAULT_N_TOP)
        self._n_top_sb.setEnabled(False)
        self._n_top_sb.setToolTip("Elements across top/bottom face")
        box_form.addRow("n_top:", self._n_top_sb)

        self._n_side_sb = QSpinBox()
        self._n_side_sb.setRange(1, 20)
        self._n_side_sb.setValue(self._DEFAULT_N_SIDE)
        self._n_side_sb.setEnabled(False)
        self._n_side_sb.setToolTip("Elements across left/right face")
        box_form.addRow("n_side:", self._n_side_sb)

        self._n_length_sb = QSpinBox()
        self._n_length_sb.setRange(1, 50)
        self._n_length_sb.setValue(self._DEFAULT_N_LENGTH)
        self._n_length_sb.setEnabled(False)
        self._n_length_sb.setToolTip("Elements along beam axis")
        box_form.addRow("n_length:", self._n_length_sb)

        box_vbox.addLayout(box_form)
        box_vbox.addStretch()
        mesh_hlayout.addWidget(box_col)

        # ── I-Profile column ─────────────────────────────────────────────────
        iprofil_col = QGroupBox("I-Profile")
        iprofil_vbox = QVBoxLayout(iprofil_col)
        iprofil_vbox.setSpacing(6)

        ip_rb_row = QHBoxLayout()
        self._iprofil_default_rb = QRadioButton("Default")
        self._iprofil_default_rb.setChecked(True)
        self._iprofil_default_rb.setToolTip(
            f"n_top={self._DEFAULT_N_TOP_I}, n_side={self._DEFAULT_N_SIDE_I}, "
            f"n_bottom={self._DEFAULT_N_BOTTOM_I}, n_length={self._DEFAULT_N_LENGTH_I}"
        )
        self._iprofil_custom_rb = QRadioButton("Custom")
        self._iprofil_default_rb.toggled.connect(self._on_iprofil_mesh_mode_changed)
        ip_rb_row.addWidget(self._iprofil_default_rb)
        ip_rb_row.addWidget(self._iprofil_custom_rb)
        iprofil_vbox.addLayout(ip_rb_row)

        ip_form = QFormLayout()
        ip_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_top_i_sb = QSpinBox()
        self._n_top_i_sb.setRange(1, 20)
        self._n_top_i_sb.setValue(self._DEFAULT_N_TOP_I)
        self._n_top_i_sb.setEnabled(False)
        self._n_top_i_sb.setToolTip("Elements across top flange width")
        ip_form.addRow("n_top:", self._n_top_i_sb)

        self._n_side_i_sb = QSpinBox()
        self._n_side_i_sb.setRange(1, 20)
        self._n_side_i_sb.setValue(self._DEFAULT_N_SIDE_I)
        self._n_side_i_sb.setEnabled(False)
        self._n_side_i_sb.setToolTip("Elements along web height")
        ip_form.addRow("n_side:", self._n_side_i_sb)

        self._n_bottom_i_sb = QSpinBox()
        self._n_bottom_i_sb.setRange(1, 20)
        self._n_bottom_i_sb.setValue(self._DEFAULT_N_BOTTOM_I)
        self._n_bottom_i_sb.setEnabled(False)
        self._n_bottom_i_sb.setToolTip("Elements across bottom flange width")
        ip_form.addRow("n_bottom:", self._n_bottom_i_sb)

        self._n_length_i_sb = QSpinBox()
        self._n_length_i_sb.setRange(1, 50)
        self._n_length_i_sb.setValue(self._DEFAULT_N_LENGTH_I)
        self._n_length_i_sb.setEnabled(False)
        self._n_length_i_sb.setToolTip("Elements along beam axis")
        ip_form.addRow("n_length:", self._n_length_i_sb)

        iprofil_vbox.addLayout(ip_form)
        iprofil_vbox.addStretch()
        mesh_hlayout.addWidget(iprofil_col)

        # ── PIPE column ──────────────────────────────────────────────────────
        pipe_col = QGroupBox("PIPE")
        pipe_vbox = QVBoxLayout(pipe_col)
        pipe_vbox.setSpacing(6)

        pipe_rb_row = QHBoxLayout()
        self._pipe_default_rb = QRadioButton("Default")
        self._pipe_default_rb.setChecked(True)
        self._pipe_default_rb.setToolTip(
            f"c_circ={self._DEFAULT_C_CIRC}, n_length={self._DEFAULT_N_LENGTH_P}"
        )
        self._pipe_custom_rb = QRadioButton("Custom")
        self._pipe_default_rb.toggled.connect(self._on_pipe_mesh_mode_changed)
        pipe_rb_row.addWidget(self._pipe_default_rb)
        pipe_rb_row.addWidget(self._pipe_custom_rb)
        pipe_vbox.addLayout(pipe_rb_row)

        pipe_form = QFormLayout()
        pipe_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._c_circ_sb = QSpinBox()
        self._c_circ_sb.setRange(3, 32)
        self._c_circ_sb.setValue(self._DEFAULT_C_CIRC)
        self._c_circ_sb.setEnabled(False)
        self._c_circ_sb.setToolTip("Elements around the circumference (≥ 3)")
        pipe_form.addRow("c_circ:", self._c_circ_sb)

        self._n_length_p_sb = QSpinBox()
        self._n_length_p_sb.setRange(1, 50)
        self._n_length_p_sb.setValue(self._DEFAULT_N_LENGTH_P)
        self._n_length_p_sb.setEnabled(False)
        self._n_length_p_sb.setToolTip("Elements along beam axis")
        pipe_form.addRow("n_length:", self._n_length_p_sb)

        pipe_vbox.addLayout(pipe_form)
        pipe_vbox.addStretch()
        mesh_hlayout.addWidget(pipe_col)

        # ── Shell column ─────────────────────────────────────────────────────
        shell_col = QGroupBox("Shell")
        shell_vbox = QVBoxLayout(shell_col)
        shell_vbox.setSpacing(6)

        shell_rb_row = QHBoxLayout()
        self._shell_default_rb = QRadioButton("Default")
        self._shell_default_rb.setChecked(True)
        self._shell_default_rb.setToolTip(
            f"mesh_12={self._DEFAULT_MESH_12}, mesh_14={self._DEFAULT_MESH_14}"
        )
        self._shell_custom_rb = QRadioButton("Custom")
        self._shell_default_rb.toggled.connect(self._on_shell_mesh_mode_changed)
        shell_rb_row.addWidget(self._shell_default_rb)
        shell_rb_row.addWidget(self._shell_custom_rb)
        shell_vbox.addLayout(shell_rb_row)

        shell_form = QFormLayout()
        shell_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._mesh_12_sb = QSpinBox()
        self._mesh_12_sb.setRange(1, 20)
        self._mesh_12_sb.setValue(self._DEFAULT_MESH_12)
        self._mesh_12_sb.setEnabled(False)
        self._mesh_12_sb.setToolTip("Elements along n1 → n2 edge direction")
        shell_form.addRow("mesh_12:", self._mesh_12_sb)

        self._mesh_14_sb = QSpinBox()
        self._mesh_14_sb.setRange(1, 20)
        self._mesh_14_sb.setValue(self._DEFAULT_MESH_14)
        self._mesh_14_sb.setEnabled(False)
        self._mesh_14_sb.setToolTip("Elements along n1 → n4 edge direction")
        shell_form.addRow("mesh_14:", self._mesh_14_sb)

        shell_vbox.addLayout(shell_form)
        shell_vbox.addStretch()
        mesh_hlayout.addWidget(shell_col)

        root.addWidget(mesh_grp)

        # ── Material thermal properties ──────────────────────────────────────
        root.addWidget(self._build_material_group())

        # ── Estimated output info ────────────────────────────────────────────
        self._est_label = QLabel()
        self._est_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._est_label)
        self._update_estimate()

        self._t_end_min.valueChanged.connect(self._update_estimate)
        self._dt_s.valueChanged.connect(self._update_estimate)
        self._out_dt_s.valueChanged.connect(self._update_estimate)

        # ── Buttons ──────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run Analysis")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ── Material group builder ────────────────────────────────────────────────

    def _build_material_group(self) -> QGroupBox:
        grp = QGroupBox("Material Thermal Properties (steel)")
        vbox = QVBoxLayout(grp)
        vbox.setSpacing(6)

        # scalar fields
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._epsilon_sb = QDoubleSpinBox()
        self._epsilon_sb.setRange(0.0, 1.0)
        self._epsilon_sb.setDecimals(2)
        self._epsilon_sb.setSingleStep(0.01)
        self._epsilon_sb.setValue(self._DEFAULT_EPSILON_STEEL)
        self._epsilon_sb.setToolTip("Steel surface emissivity ε_steel (PDF §3.2.4)")
        form.addRow("Emissivity ε:", self._epsilon_sb)

        self._density_sb = QDoubleSpinBox()
        self._density_sb.setRange(1000.0, 20000.0)
        self._density_sb.setDecimals(0)
        self._density_sb.setSuffix(" kg/m³")
        self._density_sb.setValue(self._DEFAULT_DENSITY)
        form.addRow("Density ρ:", self._density_sb)

        self._c_ref_sb = QDoubleSpinBox()
        self._c_ref_sb.setRange(1.0, 10000.0)
        self._c_ref_sb.setDecimals(1)
        self._c_ref_sb.setSuffix(" J/kg·K")
        self._c_ref_sb.setValue(self._DEFAULT_C_REF)
        self._c_ref_sb.setToolTip("Reference specific heat c_ref (multiplied by cp(T) factor table)")
        form.addRow("Base specific heat c:", self._c_ref_sb)

        self._k_ref_sb = QDoubleSpinBox()
        self._k_ref_sb.setRange(0.1, 500.0)
        self._k_ref_sb.setDecimals(1)
        self._k_ref_sb.setSuffix(" W/m·K")
        self._k_ref_sb.setValue(self._DEFAULT_K_REF)
        self._k_ref_sb.setToolTip("Reference conductivity k_ref (multiplied by k(T) factor table)")
        form.addRow("Base conductivity k:", self._k_ref_sb)

        self._enclosed_gas_sb = QDoubleSpinBox()
        self._enclosed_gas_sb.setRange(0.0, 100000.0)
        self._enclosed_gas_sb.setDecimals(0)
        self._enclosed_gas_sb.setSuffix(" J/m³·K")
        self._enclosed_gas_sb.setValue(self._DEFAULT_ENCLOSED_GAS_RHO_C)
        self._enclosed_gas_sb.setToolTip(
            "Volumetric heat capacity of enclosed gas/air inside hollow sections"
        )
        form.addRow("Enclosed-gas ρ·c:", self._enclosed_gas_sb)

        vbox.addLayout(form)

        # factor tables side by side
        tables_hlayout = QHBoxLayout()
        tables_hlayout.setSpacing(12)

        cp_vbox = QVBoxLayout()
        cp_vbox.addWidget(QLabel("cp(T) factor — multiplier on base c"))
        self._cp_table = self._make_factor_table(self._DEFAULT_CP_TABLE)
        cp_vbox.addWidget(self._cp_table)
        cp_btns = QHBoxLayout()
        cp_add = QPushButton("Add row")
        cp_add.clicked.connect(lambda: self._add_table_row(self._cp_table))
        cp_remove = QPushButton("Remove row")
        cp_remove.clicked.connect(lambda: self._remove_table_row(self._cp_table))
        cp_btns.addWidget(cp_add)
        cp_btns.addWidget(cp_remove)
        cp_vbox.addLayout(cp_btns)
        tables_hlayout.addLayout(cp_vbox)

        k_vbox = QVBoxLayout()
        k_vbox.addWidget(QLabel("k(T) factor — multiplier on base k"))
        self._k_table = self._make_factor_table(self._DEFAULT_K_TABLE)
        k_vbox.addWidget(self._k_table)
        k_btns = QHBoxLayout()
        k_add = QPushButton("Add row")
        k_add.clicked.connect(lambda: self._add_table_row(self._k_table))
        k_remove = QPushButton("Remove row")
        k_remove.clicked.connect(lambda: self._remove_table_row(self._k_table))
        k_btns.addWidget(k_add)
        k_btns.addWidget(k_remove)
        k_vbox.addLayout(k_btns)
        tables_hlayout.addLayout(k_vbox)

        vbox.addLayout(tables_hlayout)

        hint = QLabel(
            "Leave a table empty for a constant-independent property.  "
            "Temperatures must strictly increase."
        )
        hint.setStyleSheet("color: gray; font-size: 11px;")
        vbox.addWidget(hint)

        return grp

    def _make_factor_table(
        self, data: list[tuple[float, float]]
    ) -> QTableWidget:
        tbl = QTableWidget(len(data), 2)
        tbl.setHorizontalHeaderLabels(["T [°C]", "factor"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setMaximumHeight(160)
        for row, (T, f) in enumerate(data):
            tbl.setItem(row, 0, QTableWidgetItem(str(T)))
            tbl.setItem(row, 1, QTableWidgetItem(str(f)))
        return tbl

    def _add_table_row(self, tbl: QTableWidget) -> None:
        tbl.insertRow(tbl.rowCount())

    def _remove_table_row(self, tbl: QTableWidget) -> None:
        row = tbl.currentRow()
        if row >= 0:
            tbl.removeRow(row)
        elif tbl.rowCount() > 0:
            tbl.removeRow(tbl.rowCount() - 1)

    def _read_factor_table(
        self, tbl: QTableWidget
    ) -> list[tuple[float, float]]:
        rows: list[tuple[float, float]] = []
        for row in range(tbl.rowCount()):
            T_item = tbl.item(row, 0)
            f_item = tbl.item(row, 1)
            if T_item and f_item and T_item.text().strip() and f_item.text().strip():
                try:
                    rows.append((float(T_item.text()), float(f_item.text())))
                except ValueError:
                    pass
        return rows

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_dt_changed(self) -> None:
        """Keep output_dt minimum in sync with dt."""
        dt = self._dt_s.value()
        self._out_dt_s.setMinimum(dt)
        if self._out_dt_s.value() < dt:
            self._out_dt_s.setValue(dt)
        self._update_estimate()

    def _on_box_mesh_mode_changed(self) -> None:
        custom = self._box_custom_rb.isChecked()
        self._n_top_sb.setEnabled(custom)
        self._n_side_sb.setEnabled(custom)
        self._n_length_sb.setEnabled(custom)

    def _on_iprofil_mesh_mode_changed(self) -> None:
        custom = self._iprofil_custom_rb.isChecked()
        self._n_top_i_sb.setEnabled(custom)
        self._n_side_i_sb.setEnabled(custom)
        self._n_bottom_i_sb.setEnabled(custom)
        self._n_length_i_sb.setEnabled(custom)

    def _on_pipe_mesh_mode_changed(self) -> None:
        custom = self._pipe_custom_rb.isChecked()
        self._c_circ_sb.setEnabled(custom)
        self._n_length_p_sb.setEnabled(custom)

    def _on_shell_mesh_mode_changed(self) -> None:
        custom = self._shell_custom_rb.isChecked()
        self._mesh_12_sb.setEnabled(custom)
        self._mesh_14_sb.setEnabled(custom)

    def _on_accept(self) -> None:
        """Validate the config, emit signal, and close."""
        if self._n_exposed == 0:
            QMessageBox.warning(
                self,
                "No Exposed Elements",
                "There are no beam elements exposed to fire.\n\n"
                "Add at least one active fire zone before running the analysis.",
            )
            return

        try:
            cfg = self.get_config()
            cfg.validate()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Parameters", str(exc))
            return
        # Warn if a table has only one row (interpolation makes no sense)
        for label, tbl in (("cp(T)", self._cp_table), ("k(T)", self._k_table)):
            if self._read_factor_table(tbl) and len(self._read_factor_table(tbl)) == 1:
                QMessageBox.warning(
                    self, "Factor Table",
                    f"The {label} factor table has only one row — "
                    "add more points or clear it to use a constant property.",
                )
                return

        self.config_accepted.emit(cfg)
        self.accept()

    def _update_estimate(self) -> None:
        """Refresh the estimated output-step count label."""
        try:
            t_end_s   = self._t_end_min.value() * 60.0
            dt        = self._dt_s.value()
            out_dt    = self._out_dt_s.value()
            n_solver  = max(1, round(t_end_s / dt))
            n_out     = 1 + round(t_end_s / out_dt)
            self._est_label.setText(
                f"≈ {n_solver} solver steps  ·  {n_out} output snapshots stored"
            )
        except Exception:  # noqa: BLE001
            self._est_label.setText("")
