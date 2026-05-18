from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fahts.core.model.node import Node
from fahts.core.model.element import BeamElement, ShellElement
from fahts.core.model.section import Section
from fahts.core.model.material import SteelMaterial
from fahts.core.model.group import Group


@dataclass
class FEMModel:
    """
    Central data container for a USFOS structural model.
    Produced by usfos_reader.read_usfos_fem().
    Consumed by: renderer, heat solver, post-processor.
    """
    nodes:          dict[int, Node]
    elements:       dict[int, BeamElement]    # beam elements keyed by element ID
    sections:       dict[int, Section]        # keyed by geom_id
    materials:      dict[int, SteelMaterial]  # keyed by mat_id
    groups:         dict[str, Group]          # keyed by group name
    unitvecs:       dict[int, np.ndarray]     # keyed by lcoor_id, shape (3,)
    source_file:    Path
    shell_elements: dict[int, ShellElement] = field(default_factory=dict)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """Total element count: beams + shells."""
        return len(self.elements) + len(self.shell_elements)

    def node_array(self) -> np.ndarray:
        """Return (N, 3) array of all node coordinates, ordered by node ID."""
        nids = sorted(self.nodes.keys())
        return np.array([[self.nodes[n].x, self.nodes[n].y, self.nodes[n].z]
                         for n in nids])

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) of the model."""
        coords = self.node_array()
        return coords.min(axis=0), coords.max(axis=0)

    def centroid(self) -> np.ndarray:
        """Return geometric centroid of all nodes."""
        return self.node_array().mean(axis=0)

    def elements_in_group(self, group_name: str) -> list[BeamElement]:
        """Return BeamElement objects belonging to a named group."""
        if group_name not in self.groups:
            return []
        return [self.elements[eid]
                for eid in self.groups[group_name].element_ids
                if eid in self.elements]

    def shell_elements_in_group(self, group_name: str) -> list[ShellElement]:
        """Return ShellElement objects belonging to a named group."""
        if group_name not in self.groups:
            return []
        return [self.shell_elements[eid]
                for eid in self.groups[group_name].element_ids
                if eid in self.shell_elements]

    def summary(self) -> str:
        bb_min, bb_max = self.bounding_box()
        dims = bb_max - bb_min
        from fahts.core.model.section import BoxSection, PipeSection, ISection, PlateSection
        sec_counts: dict[str, int] = {}
        for s in self.sections.values():
            t = (type(s).__name__
                 .replace("BoxSection", "BOX")
                 .replace("PipeSection", "PIPE")
                 .replace("ISection", "IHPROFIL")
                 .replace("PlateSection", "PLTHICK"))
            sec_counts[t] = sec_counts.get(t, 0) + 1
        sec_summary = ", ".join(f"{v} {k}" for k, v in sorted(sec_counts.items()))
        return (
            f"FEMModel: {self.n_nodes} nodes, "
            f"{len(self.elements)} beams, {len(self.shell_elements)} shells\n"
            f"  Sections : {len(self.sections)} ({sec_summary})\n"
            f"  Materials: {len(self.materials)}\n"
            f"  Groups   : {len(self.groups)} — {', '.join(sorted(self.groups))}\n"
            f"  Bounds   : X [{bb_min[0]:.2f}, {bb_max[0]:.2f}] m  "
            f"Y [{bb_min[1]:.2f}, {bb_max[1]:.2f}] m  "
            f"Z [{bb_min[2]:.2f}, {bb_max[2]:.2f}] m\n"
            f"  Size     : {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} m\n"
            f"  Source   : {self.source_file}"
        )
