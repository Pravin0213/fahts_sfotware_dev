"""VessFire input deck reader (compatibility re-export).

The reader lives in the product as an importer: ``fahts.coupling.vessfire_import``.
"""

from fahts.coupling.vessfire_import import read_deck as read_case

__all__ = ["read_case"]
