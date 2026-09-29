from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    nid: int
    x: float
    y: float
    z: float
    bc: tuple[int, ...] = field(default_factory=tuple)  # boundary code flags

    @property
    def xyz(self):
        import numpy as np
        return np.array([self.x, self.y, self.z], dtype=float)
