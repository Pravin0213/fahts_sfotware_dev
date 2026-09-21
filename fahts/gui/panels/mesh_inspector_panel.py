"""
Mesh Inspector panel — shows FEM surface mesh connectivity for a clicked quad.

Click any quad on the inspector overlay to see:
  - Which beam element it belongs to and its local quad index
  - The four local node indices for that quad
  - For each node: every other quad in the same mesh that shares it
    (shared nodes == shared DOFs == non-zero K-matrix coupling)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MeshInspectorPanel(QWidget):
    """
    Sidebar panel showing FEM mesh topology for a selected quad element.

    Call ``show_quad_info(beam_eid, quad_idx, quad_nodes, beam_quads)`` when the
    user clicks a quad in the viewport; call ``clear()`` to reset to placeholder.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def show_quad_info(
        self,
        beam_eid: int,
        quad_idx: int,
        quad_nodes: list[int],
        beam_quads: np.ndarray,
        quad_gdofs: list[int] | None = None,
    ) -> None:
        """
        Populate the panel for the selected quad.

        Parameters
        ----------
        beam_eid    : element ID of the owning beam/shell.
        quad_idx    : local quad index within that element's surface mesh.
        quad_nodes  : list of 4 local node indices for this quad (used for
                      intra-element K-neighbour lookup).
        beam_quads  : (n_quads, 4) full quad-node connectivity for this element.
        quad_gdofs  : list of 4 global DOF indices for this quad (solver K-assembly
                      numbering).  Shown in the Node column when provided.
        """
        n_quads = len(beam_quads)
        display = quad_gdofs if quad_gdofs is not None else quad_nodes
        nodes_str = "  ".join(str(n) for n in display)
        self._info_label.setText(
            f"Beam EID : {beam_eid}\n"
            f"Quad     : {quad_idx}  (of {n_quads})\n"
            f"Nodes    : {nodes_str}"
        )

        # Build connectivity table: for each of the 4 nodes find all OTHER quads
        # in this beam that contain that node (K-matrix intra-element neighbours).
        self._table.setRowCount(4)
        for row, (local_n, disp_n) in enumerate(zip(quad_nodes, display)):
            sharing = [
                q for q in range(n_quads)
                if q != quad_idx and local_n in beam_quads[q]
            ]
            self._table.setItem(row, 0, QTableWidgetItem(str(disp_n)))
            share_str = ", ".join(str(q) for q in sharing) if sharing else "—"
            self._table.setItem(row, 1, QTableWidgetItem(share_str))

        self._table.resizeColumnToContents(0)

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._info_label.setText("Click a mesh quad to inspect")
        self._table.setRowCount(0)

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("Mesh Inspector")
        title.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(title)

        self._info_label = QLabel("Click a mesh quad to inspect")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #cccccc;"
        )
        layout.addWidget(self._info_label)

        # Connectivity table: Node | Shared quads
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Node", "Shared quads (K neighbours)"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setStyleSheet("font-family: monospace; font-size: 10px;")
        self._table.setMaximumHeight(140)
        layout.addWidget(self._table)

        layout.addStretch()
