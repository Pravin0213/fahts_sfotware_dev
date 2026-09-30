"""FAHTS desktop GUI (PyQt6)."""

import os

# pyvistaqt uses qtpy, which picks the first Qt binding already imported and otherwise
# defaults to PyQt5 (also installed on some machines). Pin it to PyQt6, or VTK widgets get
# a different binding from ours and cannot be parented to our widgets.
os.environ.setdefault("QT_API", "pyqt6")
