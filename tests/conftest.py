"""
Global pytest configuration.
Sets QT_QPA_PLATFORM=offscreen before any Qt imports so PyQt6 and pyvistaqt
work on headless CI / development servers without a display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Suppress VTK OpenGL warnings on headless hosts — they do not affect correctness.
os.environ.setdefault("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", "1")
