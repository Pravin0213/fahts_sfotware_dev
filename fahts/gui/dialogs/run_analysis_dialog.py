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
    QLabel,
    QMessageBox,
    QRadioButton,
    QSpinBox,
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
