from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BeamElement:
    eid: int
    n1: int           # start node ID
    n2: int           # end node ID
    mat_id: int
    geom_id: int      # references a Section (BoxSection, PipeSection, ISection)
    lcoor_id: int     # references a UNITVEC local z-axis

    # Derived fields — populated by the reader after all nodes are known
    length: float = 0.0
    direction: Optional[np.ndarray] = field(default=None, repr=False)  # unit vec n1→n2
    local_z: Optional[np.ndarray]   = field(default=None, repr=False)  # UNITVEC
    # Eccentricity offsets in global coordinates [m] (from USFOS ECCENT card)
    # The actual beam end positions are node_pos + ecc (None means no offset)
    ecc1: Optional[np.ndarray] = field(default=None, repr=False)  # offset at n1
    ecc2: Optional[np.ndarray] = field(default=None, repr=False)  # offset at n2

    def midpoint(self, nodes: dict) -> np.ndarray:
        """Return the 3-D midpoint between n1 and n2."""
        p1 = nodes[self.n1].xyz
        p2 = nodes[self.n2].xyz
        return 0.5 * (p1 + p2)


@dataclass
class ShellElement:
    """
    Flat shell element — USFOS QUADSHEL (4 nodes) or TRISHELL (3 nodes) card.

    nodes : tuple of 3 or 4 node IDs defining the element polygon.
    """
    eid: int
    nodes: tuple[int, ...]  # 3 nodes (TRISHELL) or 4 nodes (QUADSHEL)
    mat_id: int
    geom_id: int  # references a PlateSection
