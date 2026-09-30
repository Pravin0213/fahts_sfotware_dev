#!/usr/bin/env python3
"""
FAHTS — Fire Analysis and Heat Transfer Software
GUI entry point.

Usage
-----
    python -m fahts                                  # open empty application
    python -m fahts examples/models/model_file.fem   # open with model pre-loaded
    python -m fahts my_case.vcase.json               # open a vessel case (process tab)
    python -m fahts path/to/vessfire_deck_folder     # import a VessFire input deck
    fahts examples/models/model_file.fem             # same, after pip install -e .
"""

import contextlib
import ctypes
import os
import platform
import sys
from pathlib import Path


# ── Mesa GLX fix for Anaconda / bundled Python environments ──────────────────
#
# Anaconda ships its own libstdc++.so.6, which is often older than the system
# one.  Mesa's swrast DRI driver (swrast_dri.so) requires GLIBCXX_3.4.30,
# which is present in the system libstdc++ but absent from Anaconda's copy.
#
# When Mesa's DRI loader calls dlopen(swrast_dri.so) it fails because the
# Anaconda libstdc++ is already loaded and lacks the required symbol version.
# VTK then falls back to vtkEGLRenderWindow, which renders off-screen and
# produces a blank (transparent) viewport in the Qt widget.
#
# Fix: force-load the system libstdc++.so.6 with RTLD_GLOBAL *before* any VTK
# import so that the system version wins the symbol-version race.  This is safe
# on non-Anaconda systems too — if the system lib is not newer, dlopen just
# returns the already-loaded handle.

if platform.system() == "Linux":
    _sys_stdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if os.path.exists(_sys_stdcxx):
        try:
            ctypes.CDLL(_sys_stdcxx, ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    del _sys_stdcxx


# ── Silence VTK / Mesa noise ──────────────────────────────────────────────────
#
# On systems without a hardware GPU (or with missing Mesa DRI drivers), VTK
# falls back to a software EGL renderer.  Two sources of noise appear:
#
#   1. "libGL error: MESA-LOADER: failed to open iris/swrast" — emitted by
#      Mesa's DRI loader directly to the C-level fd 2 (not via Python's
#      sys.stderr). Appears once when the first render window is created.
#
#   2. "WARN| vtkEGLRenderWindow: Failed to initialize OpenGL functions!" —
#      emitted via VTK's observer system every ~0.2 s by pyvistaqt's render
#      timer.  This floods the terminal while the app is open.
#
# Fix (1): temporarily redirect fd 2 to /dev/null during window construction.
# Fix (2): vtkObject.GlobalWarningDisplayOff() before any VTK-related import.
#
# Both suppressions are safe: the app renders correctly despite these messages
# because VTK ultimately finds a working (software) fallback renderer.

import vtk as _vtk          # must import before pyvistaqt to call GlobalWarningDisplayOff
_vtk.vtkObject.GlobalWarningDisplayOff()
del _vtk


@contextlib.contextmanager
def _no_gl_noise():
    """Temporarily redirect the OS-level stderr during GL driver probing."""
    if platform.system() != "Linux":
        yield
        return
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(null_fd)
        os.close(saved_fd)


# ── Uncaught exceptions ───────────────────────────────────────────────────────
#
# PyQt6 aborts the whole application (qFatal) on an exception escaping a slot, so any bug
# in a callback would lose the user's unsaved work. Log it and show it instead.

def _install_exception_hook() -> None:
    import logging
    import traceback

    from PyQt6.QtWidgets import QMessageBox

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("fahts").error("Unhandled exception:\n%s", text)
        box = QMessageBox(QMessageBox.Icon.Critical, "Unexpected error",
                          f"{exc_type.__name__}: {exc}\n\nThe application keeps running; "
                          "please save your work and report this error.")
        box.setDetailedText(text)
        box.exec()

    sys.excepthook = hook


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    fem_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    try:
        from PyQt6.QtWidgets import QApplication
        from fahts.gui.main_window import MainWindow
    except ImportError as exc:
        print(f"[FAHTS] GUI dependencies missing: {exc}")
        print('  Run:  pip install -e ".[gui]"')
        sys.exit(1)

    app = QApplication(sys.argv)
    _install_exception_hook()
    app.setApplicationName("FAHTS")
    app.setApplicationVersion("0.1")

    with _no_gl_noise():
        window = MainWindow()

    window.show()

    if fem_file:
        # a vessel case (*.json) or a VessFire deck folder opens in the process workspace
        if fem_file.is_dir() or fem_file.suffix.lower() == ".json":
            window.open_process_case(fem_file)
        else:
            window.open_file(fem_file)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
