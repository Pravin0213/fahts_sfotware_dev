"""
Global pytest configuration.
Sets QT_QPA_PLATFORM=offscreen before any Qt imports so PyQt6 and pyvistaqt
work on headless CI / development servers without a display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
