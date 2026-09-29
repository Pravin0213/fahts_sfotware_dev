#!/usr/bin/env python3
"""Convenience launcher: ``python main.py [model.fem]`` without installing the package.

The application lives in ``src/fahts/__main__.py``; prefer ``python -m fahts`` after
``pip install -e ".[gui]"``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fahts.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
