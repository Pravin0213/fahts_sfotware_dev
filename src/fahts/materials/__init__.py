"""Temperature-dependent material properties (thermal and strength)."""

from fahts.materials.en1993_carbon_steel import en1993_carbon_steel
from fahts.materials.steel_table import SteelTable

__all__ = ["SteelTable", "en1993_carbon_steel"]
