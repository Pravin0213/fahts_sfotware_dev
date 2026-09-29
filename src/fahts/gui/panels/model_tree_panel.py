"""
Phase 1.7 — model_tree_panel.py
QTreeWidget listing model structure, named groups (with visibility checkboxes),
and materials.  Connects to SceneManager for live group show/hide.

Tree layout
-----------
  ▼ Structure
      Nodes · 659
      Elements · 783
    ▼ Groups · 17
        ☑ ELEV_A · 42 elements
        ☑ ELEV_B · 42 elements
        …
  ▼ Materials
      [1] S355  E=210 GPa  fy=355 MPa  ρ=7850 kg/m³
  ▼ Heat Sources
      (none — add fire zone in Phase 2)
"""
from __future__ import annotations

import logging
from typing import Generator, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fahts.core.model.fem_model import FEMModel

log = logging.getLogger(__name__)

# UserRole data tags stored on each tree item: ("group", name), ("material", mid) …
_KIND_GROUP    = "group"
_KIND_MATERIAL = "material"


class ModelTreePanel(QWidget):
    """
    Left-panel tree listing FAHTS model components.

    Signals
    -------
    group_visibility_changed(group_name: str, visible: bool)
        Emitted when a group checkbox is toggled.
    group_selected(group_name: str)
        Emitted when a group item is single-clicked (for properties panel).
    """

    group_visibility_changed: pyqtSignal = pyqtSignal(str, bool)
    group_selected: pyqtSignal = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = None          # SceneManager set via set_scene()
        self._model: FEMModel | None = None
        self._populating: bool = False   # guards itemChanged during populate()

        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_scene(self, scene) -> None:
        """
        Attach a SceneManager.  Group visibility changes will be forwarded to it.

        Parameters
        ----------
        scene : SceneManager
        """
        self._scene = scene

    def populate(self, model: FEMModel) -> None:
        """
        Fill the tree from *model*.  Clears previous content first.

        Safe to call multiple times (e.g. when a new file is opened).
        """
        self._model = model
        self._populating = True
        try:
            self._tree.clear()
            self._populate_structure(model)
            self._populate_materials(model)
            self._populate_heat_sources()
            self._tree.expandAll()
        finally:
            self._populating = False

        self._btn_show_all.setEnabled(True)
        self._btn_hide_all.setEnabled(True)
        log.debug("ModelTreePanel populated: %d groups, %d materials",
                  len(model.groups), len(model.materials))

    def checked_groups(self) -> list[str]:
        """Return the names of groups whose checkbox is currently ticked."""
        result: list[str] = []
        for item in self._iter_items():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == _KIND_GROUP:
                if item.checkState(0) == Qt.CheckState.Checked:
                    result.append(data[1])
        return result

    def show_all_groups(self) -> None:
        """Check all group checkboxes and make all groups visible."""
        self._set_all_group_checkboxes(Qt.CheckState.Checked)
        if self._scene is not None:
            self._scene.set_all_groups_visibility(True)

    def hide_all_groups(self) -> None:
        """Uncheck all group checkboxes and hide all groups."""
        self._set_all_group_checkboxes(Qt.CheckState.Unchecked)
        if self._scene is not None:
            self._scene.set_all_groups_visibility(False)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Show / Hide all buttons
        btn_row = QHBoxLayout()
        self._btn_show_all = QPushButton("Show All")
        self._btn_hide_all = QPushButton("Hide All")
        self._btn_show_all.setEnabled(False)
        self._btn_hide_all.setEnabled(False)
        self._btn_show_all.clicked.connect(self.show_all_groups)
        self._btn_hide_all.clicked.connect(self.hide_all_groups)
        btn_row.addWidget(self._btn_show_all)
        btn_row.addWidget(self._btn_hide_all)
        layout.addLayout(btn_row)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Model")
        self._tree.setColumnCount(1)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    # ── Tree population ───────────────────────────────────────────────────────

    def _populate_structure(self, model: FEMModel) -> None:
        struct = QTreeWidgetItem(self._tree, ["Structure"])
        struct.setExpanded(True)
        _make_leaf(struct, f"Nodes  ·  {model.n_nodes}")
        _make_leaf(struct, f"Elements  ·  {model.n_elements}")

        groups_root = QTreeWidgetItem(struct, [f"Groups  ·  {len(model.groups)}"])
        groups_root.setExpanded(True)

        for name in sorted(model.groups.keys()):
            n_elem = len(model.groups[name])
            item = QTreeWidgetItem(groups_root, [f"{name}  ·  {n_elem} elements"])
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, (_KIND_GROUP, name))

    def _populate_materials(self, model: FEMModel) -> None:
        mat_root = QTreeWidgetItem(self._tree, ["Materials"])
        mat_root.setExpanded(True)
        for mid, mat in sorted(model.materials.items()):
            label = (
                f"[{mid}] {mat.name}  ·  "
                f"E={mat.E/1e9:.0f} GPa  "
                f"fy={mat.fy/1e6:.0f} MPa  "
                f"ρ={mat.rho:.0f} kg/m³"
            )
            item = _make_leaf(mat_root, label)
            item.setData(0, Qt.ItemDataRole.UserRole, (_KIND_MATERIAL, mid))

    def _populate_heat_sources(self) -> None:
        hs_root = QTreeWidgetItem(self._tree, ["Heat Sources"])
        hs_root.setExpanded(True)
        placeholder = _make_leaf(hs_root, "(none — add fire zone in Phase 2)")
        placeholder.setDisabled(True)

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._populating:
            return
        data = item.data(column, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, value = data
        if kind == _KIND_GROUP:
            visible = item.checkState(column) == Qt.CheckState.Checked
            self.group_visibility_changed.emit(value, visible)
            if self._scene is not None:
                self._scene.set_group_visibility(value, visible)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(column, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, value = data
        if kind == _KIND_GROUP:
            self.group_selected.emit(value)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_all_group_checkboxes(self, state: Qt.CheckState) -> None:
        """Set all group item checkboxes to *state* without triggering per-item callbacks."""
        self._populating = True
        try:
            for item in self._iter_items():
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if data and data[0] == _KIND_GROUP:
                    item.setCheckState(0, state)
        finally:
            self._populating = False

    def _iter_items(self) -> Generator[QTreeWidgetItem, None, None]:
        """Depth-first iteration over all QTreeWidgetItems."""
        stack = [self._tree.invisibleRootItem()]
        while stack:
            parent = stack.pop()
            for i in range(parent.childCount()):
                child = parent.child(i)
                yield child
                stack.append(child)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_leaf(parent: QTreeWidgetItem, text: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem(parent, [text])
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
    return item
