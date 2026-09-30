"""Architecture check: physics packages only import the layers below them.

Keeps each part separable and debuggable on its own. Only ``coupling`` may combine
packages; see src/fahts/coupling/CLAUDE.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "fahts"

ALLOWED = {
    "common": set(),
    "thermo": {"common"},
    "materials": {"common"},
    "fire": {"common"},
    "wall": {"materials", "common", "core"},        # fem_3d uses the Hex8 kernel in core.heat
    "relief": {"thermo", "common"},
    "process": {"thermo", "common"},
    "rupture": {"materials", "common"},
    "coupling": {"common", "thermo", "materials", "fire", "wall", "relief", "process",
                 "rupture"},
}


def _fahts_imports(path: Path) -> set[str]:
    """Top-level fahts sub-packages imported by one file."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for n in names:
            parts = n.split(".")
            if parts[0] == "fahts" and len(parts) > 1:
                out.add(parts[1])
    return out


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_imports_respect_layers(package):
    bad = {}
    for f in sorted((SRC / package).rglob("*.py")):
        extra = _fahts_imports(f) - ALLOWED[package] - {package}
        if extra:
            bad[str(f.relative_to(SRC))] = sorted(extra)
    assert not bad, f"layering violation (see coupling/CLAUDE.md): {bad}"
