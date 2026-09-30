"""
Global pytest configuration.
Sets QT_QPA_PLATFORM=offscreen before any Qt imports so PyQt6 and pyvistaqt
work on headless CI / development servers without a display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyqt6")   # see fahts/gui/__init__.py

# Suppress VTK OpenGL warnings on headless hosts — they do not affect correctness.
os.environ.setdefault("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", "1")


# ── Golden regression tests (process model) ──────────────────────────────────
# Opt-in: they re-run full vessel simulations (~4 min). Run with --golden whenever
# code ported from legacy/vfpy changes: python -m pytest tests/regression --golden -q

def pytest_addoption(parser):
    parser.addoption("--golden", action="store_true", default=False,
                     help="run golden regression tests against frozen reference outputs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--golden"):
        return
    import pytest
    skip = pytest.mark.skip(reason="golden regression test: run with --golden")
    for item in items:
        if "golden" in item.keywords:
            item.add_marker(skip)


# ── Synthetic steel table (no proprietary data) for process-model unit/integration tests ──
import numpy as _np  # noqa: E402
import pytest as _pytest  # noqa: E402

from fahts.materials import SteelTable  # noqa: E402


@_pytest.fixture
def steel() -> SteelTable:
    """Synthetic carbon-steel-like table (no proprietary data): cp and k vary with T."""
    T = _np.array([273.15, 473.15, 673.15, 873.15, 1003.15, 1073.15, 1273.15, 1473.15])
    return SteelTable(
        name="synthetic",
        T=T,
        cp=_np.array([440.0, 530.0, 606.0, 760.0, 5000.0, 800.0, 650.0, 650.0]),
        k=_np.array([54.0, 48.0, 41.0, 34.0, 29.0, 27.0, 27.0, 27.0]),
        rho=7850.0,
        f_yield=_np.array([1.0, 1.0, 0.8, 0.47, 0.2, 0.11, 0.04, 0.0]),
        f_uts=_np.array([1.0, 1.0, 1.0, 0.47, 0.2, 0.11, 0.04, 0.0]),
    )


@_pytest.fixture
def legacy_steel(steel):
    """The same table as the legacy vfpy Material class."""
    from tests.legacy_ref import legacy
    ht = legacy("heat_transfer")
    return ht.Material(steel.name, steel.T, steel.cp, steel.k, steel.rho, steel.f_yield,
                       steel.f_uts)

