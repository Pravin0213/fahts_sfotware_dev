"""
MeshPreviewDialog — configure FEM mesh parameters and preview on the structure.

No fire zones are required.  Emits mesh_accepted(AnalysisConfig) on OK so the
caller can build the analysis mesh overlay immediately.

The AnalysisConfig returned carries only mesh parameters; t_end/dt/output_dt
are set to sensible defaults and do not affect the preview.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fahts.core.results.analysis_config import AnalysisConfig


class MeshPreviewDialog(QDialog):
    """
    Modal dialog for setting FEM mesh parameters to preview on the structure.

    Emits ``mesh_accepted(AnalysisConfig)`` when the user clicks Apply.
    The returned config carries only mesh parameters; time-integration fields
    are filled with defaults and should be ignored by the preview code.
    """

    mesh_accepted: pyqtSignal = pyqtSignal(object)   # emits AnalysisConfig

    # Defaults match RunAnalysisDialog and FAHTS software defaults
    _DEFAULT_N_TOP:    int = 2
    _DEFAULT_N_SIDE:   int = 3
    _DEFAULT_N_LENGTH: int = 4

    _DEFAULT_N_TOP_I:    int = 4
    _DEFAULT_N_SIDE_I:   int = 2
    _DEFAULT_N_BOTTOM_I: int = 2
    _DEFAULT_N_LENGTH_I: int = 2

    _DEFAULT_C_CIRC:    int = 8
    _DEFAULT_N_LENGTH_P: int = 4

    _DEFAULT_MESH_12: int = 4
    _DEFAULT_MESH_14: int = 2

    def __init__(
        self,
        parent: QWidget | None = None,
        current_config: AnalysisConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self._current = current_config
        self.setWindowTitle("Create Mesh Preview")
        self.setMinimumWidth(700)
        self._build_ui()
        if current_config is not None:
            self._populate_from_config(current_config)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_config(self) -> AnalysisConfig:
        """Return an AnalysisConfig built from the current field values."""
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
            t_end=7200.0, dt=60.0, output_dt=60.0,   # dummy time params
            n_top=n_top, n_side=n_side, n_length=n_length,
            n_top_i=n_top_i, n_side_i=n_side_i,
            n_bottom_i=n_bottom_i, n_length_i=n_length_i,
            c_circ=c_circ, n_length_p=n_length_p,
            mesh_12=mesh_12, mesh_14=mesh_14,
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        info = QLabel(
            "Configure the FEM mesh density for each section type.  "
            "The mesh will be shown on the structure as a wireframe overlay."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # ── Four-column mesh groups ───────────────────────────────────────────
        mesh_grp = QGroupBox("Mesh Parameters")
        hlayout = QHBoxLayout(mesh_grp)
        hlayout.setSpacing(8)

        # BOX ──────────────────────────────────────────────────────────────────
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
        self._box_default_rb.toggled.connect(self._on_box_mode)
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
        hlayout.addWidget(box_col)

        # I-Profile ────────────────────────────────────────────────────────────
        iprofil_col = QGroupBox("I-Profile")
        ip_vbox = QVBoxLayout(iprofil_col)
        ip_vbox.setSpacing(6)

        ip_rb_row = QHBoxLayout()
        self._iprofil_default_rb = QRadioButton("Default")
        self._iprofil_default_rb.setChecked(True)
        self._iprofil_default_rb.setToolTip(
            f"n_top={self._DEFAULT_N_TOP_I}, n_side={self._DEFAULT_N_SIDE_I}, "
            f"n_bottom={self._DEFAULT_N_BOTTOM_I}, n_length={self._DEFAULT_N_LENGTH_I}"
        )
        self._iprofil_custom_rb = QRadioButton("Custom")
        self._iprofil_default_rb.toggled.connect(self._on_iprofil_mode)
        ip_rb_row.addWidget(self._iprofil_default_rb)
        ip_rb_row.addWidget(self._iprofil_custom_rb)
        ip_vbox.addLayout(ip_rb_row)

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

        ip_vbox.addLayout(ip_form)
        ip_vbox.addStretch()
        hlayout.addWidget(iprofil_col)

        # PIPE ─────────────────────────────────────────────────────────────────
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
        self._pipe_default_rb.toggled.connect(self._on_pipe_mode)
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
        hlayout.addWidget(pipe_col)

        # Shell ────────────────────────────────────────────────────────────────
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
        self._shell_default_rb.toggled.connect(self._on_shell_mode)
        shell_rb_row.addWidget(self._shell_default_rb)
        shell_rb_row.addWidget(self._shell_custom_rb)
        shell_vbox.addLayout(shell_rb_row)

        shell_form = QFormLayout()
        shell_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._mesh_12_sb = QSpinBox()
        self._mesh_12_sb.setRange(1, 20)
        self._mesh_12_sb.setValue(self._DEFAULT_MESH_12)
        self._mesh_12_sb.setEnabled(False)
        self._mesh_12_sb.setToolTip("Elements along n1→n2 edge direction")
        shell_form.addRow("mesh_12:", self._mesh_12_sb)

        self._mesh_14_sb = QSpinBox()
        self._mesh_14_sb.setRange(1, 20)
        self._mesh_14_sb.setValue(self._DEFAULT_MESH_14)
        self._mesh_14_sb.setEnabled(False)
        self._mesh_14_sb.setToolTip("Elements along n1→n4 edge direction")
        shell_form.addRow("mesh_14:", self._mesh_14_sb)

        shell_vbox.addLayout(shell_form)
        shell_vbox.addStretch()
        hlayout.addWidget(shell_col)

        root.addWidget(mesh_grp)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Mesh")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_box_mode(self) -> None:
        custom = self._box_custom_rb.isChecked()
        self._n_top_sb.setEnabled(custom)
        self._n_side_sb.setEnabled(custom)
        self._n_length_sb.setEnabled(custom)

    def _on_iprofil_mode(self) -> None:
        custom = self._iprofil_custom_rb.isChecked()
        self._n_top_i_sb.setEnabled(custom)
        self._n_side_i_sb.setEnabled(custom)
        self._n_bottom_i_sb.setEnabled(custom)
        self._n_length_i_sb.setEnabled(custom)

    def _on_pipe_mode(self) -> None:
        custom = self._pipe_custom_rb.isChecked()
        self._c_circ_sb.setEnabled(custom)
        self._n_length_p_sb.setEnabled(custom)

    def _on_shell_mode(self) -> None:
        custom = self._shell_custom_rb.isChecked()
        self._mesh_12_sb.setEnabled(custom)
        self._mesh_14_sb.setEnabled(custom)

    def _on_accept(self) -> None:
        cfg = self.get_config()
        self.mesh_accepted.emit(cfg)
        self.accept()

    def _populate_from_config(self, cfg: AnalysisConfig) -> None:
        """Pre-fill spinboxes from an existing config (e.g. from a prior preview)."""
        # BOX
        defaults_box = (
            cfg.n_top == self._DEFAULT_N_TOP
            and cfg.n_side == self._DEFAULT_N_SIDE
            and cfg.n_length == self._DEFAULT_N_LENGTH
        )
        if not defaults_box:
            self._box_custom_rb.setChecked(True)
            self._n_top_sb.setValue(cfg.n_top)
            self._n_side_sb.setValue(cfg.n_side)
            self._n_length_sb.setValue(cfg.n_length)

        # I-Profile
        defaults_ip = (
            cfg.n_top_i == self._DEFAULT_N_TOP_I
            and cfg.n_side_i == self._DEFAULT_N_SIDE_I
            and cfg.n_bottom_i == self._DEFAULT_N_BOTTOM_I
            and cfg.n_length_i == self._DEFAULT_N_LENGTH_I
        )
        if not defaults_ip:
            self._iprofil_custom_rb.setChecked(True)
            self._n_top_i_sb.setValue(cfg.n_top_i)
            self._n_side_i_sb.setValue(cfg.n_side_i)
            self._n_bottom_i_sb.setValue(cfg.n_bottom_i)
            self._n_length_i_sb.setValue(cfg.n_length_i)

        # PIPE
        if cfg.c_circ != self._DEFAULT_C_CIRC or cfg.n_length_p != self._DEFAULT_N_LENGTH_P:
            self._pipe_custom_rb.setChecked(True)
            self._c_circ_sb.setValue(cfg.c_circ)
            self._n_length_p_sb.setValue(cfg.n_length_p)

        # Shell
        if cfg.mesh_12 != self._DEFAULT_MESH_12 or cfg.mesh_14 != self._DEFAULT_MESH_14:
            self._shell_custom_rb.setChecked(True)
            self._mesh_12_sb.setValue(cfg.mesh_12)
            self._mesh_14_sb.setValue(cfg.mesh_14)
