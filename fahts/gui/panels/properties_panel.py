"""
Phase 1.8 — properties_panel.py
Displays structural properties of the selected beam element.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fahts.core.model.fem_model import FEMModel
from fahts.core.model.section import BoxSection


class PropertiesPanel(QWidget):
    """
    Sidebar panel showing structural properties of a selected beam element.

    Call ``show_element(eid, model)`` when a beam is clicked in the viewport;
    call ``clear()`` to return to the placeholder state.

    Signals
    -------
    element_selected(int)
        Emitted after ``show_element`` finishes populating the panel.
        The int is the element ID that was displayed.
    """

    element_selected: pyqtSignal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_eid: int | None = None
        self._init_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def show_element(self, eid: int, model: FEMModel) -> None:
        """Populate the panel with properties of element *eid* from *model*."""
        if eid not in model.elements:
            self.clear()
            return

        elem = model.elements[eid]
        self._current_eid = eid

        groups = sorted(
            name for name, g in model.groups.items()
            if eid in g.element_ids
        )
        section  = model.sections.get(elem.geom_id)
        material = model.materials.get(elem.mat_id)

        self._clear_content()

        self._add_header("Element")
        self._add_row("ID:", str(eid))
        self._add_row("Length:", f"{elem.length:.3f} m")
        self._add_row("Nodes:", f"{elem.n1} → {elem.n2}")
        if groups:
            self._add_row("Groups:", ", ".join(groups))

        if isinstance(section, BoxSection):
            self._add_divider()
            self._add_header(f"Section  (BOX {section.sid})")
            self._add_row("Height H:", f"{section.H * 1e3:.1f} mm")
            self._add_row("Width W:", f"{section.W * 1e3:.1f} mm")
            self._add_row("T_side:", f"{section.T_side * 1e3:.1f} mm")
            self._add_row("T_bot:", f"{section.T_bot * 1e3:.1f} mm")
            self._add_row("T_top:", f"{section.T_top * 1e3:.1f} mm")
            self._add_row("Steel area:", f"{section.cross_section_area * 1e6:.0f} mm²")
            self._add_row("Perimeter:", f"{section.outer_perimeter * 1e3:.0f} mm")
            self._add_row("Am/V:", f"{section.section_factor_Am_V:.1f} m⁻¹")

        if material is not None:
            self._add_divider()
            self._add_header(f"Material  ({material.name})")
            self._add_row("E:", f"{material.E / 1e9:.0f} GPa")
            self._add_row("fy:", f"{material.fy / 1e6:.0f} MPa")
            self._add_row("ρ:", f"{material.rho:.0f} kg/m³")
            self._add_row("αT:", f"{material.alpha_T:.2e} /K")

        self._content_layout.addStretch()
        self._placeholder.hide()
        self._scroll_area.show()

        self.element_selected.emit(eid)

    def clear(self) -> None:
        """Return to the no-selection placeholder state."""
        self._current_eid = None
        self._clear_content()
        self._scroll_area.hide()
        self._placeholder.show()

    @property
    def current_eid(self) -> int | None:
        """The element ID currently displayed, or None if nothing is selected."""
        return self._current_eid

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        hdr = QLabel("Properties")
        hdr.setStyleSheet("font-weight: bold; color: #aaaaaa;")
        outer.addWidget(hdr)

        self._placeholder = QLabel("Click a beam element\nto view its properties.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._placeholder.setStyleSheet("color: #666666; font-style: italic;")
        outer.addWidget(self._placeholder)

        # Detail area — hidden until show_element() is called
        self._scroll_area = QScrollArea()
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll_area.hide()

        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 4, 0)
        self._content_layout.setSpacing(2)
        self._scroll_area.setWidget(self._content_widget)

        outer.addWidget(self._scroll_area, 1)

    # ── Content helpers ───────────────────────────────────────────────────────

    def _clear_content(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_header(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #cccccc; margin-top: 4px;")
        self._content_layout.addWidget(lbl)

    def _add_divider(self) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #444444; margin: 3px 0;")
        self._content_layout.addWidget(line)

    def _add_row(self, label: str, value: str) -> None:
        """Add a label-value pair row."""
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(2, 1, 2, 1)
        hl.setSpacing(4)

        key = QLabel(label)
        key.setStyleSheet("color: #888888; font-size: 11px;")
        key.setFixedWidth(72)
        key.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        val = QLabel(value)
        val.setStyleSheet("color: #dddddd; font-size: 11px;")
        val.setWordWrap(True)

        hl.addWidget(key)
        hl.addWidget(val, 1)

        self._content_layout.addWidget(container)
