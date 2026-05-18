from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Group:
    """Named set of element IDs (from USFOS Name/GroupDef cards)."""
    gid: int
    name: str
    element_ids: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.element_ids)

    def __contains__(self, eid: int) -> bool:
        return eid in self.element_ids
