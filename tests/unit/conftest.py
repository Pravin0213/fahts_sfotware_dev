"""Shared fixtures for unit tests of the ported process-model packages."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.materials import SteelTable


@pytest.fixture
def steel() -> SteelTable:
    """Synthetic carbon-steel-like table (no proprietary data): cp and k vary with T."""
    T = np.array([273.15, 473.15, 673.15, 873.15, 1003.15, 1073.15, 1273.15, 1473.15])
    return SteelTable(
        name="synthetic",
        T=T,
        cp=np.array([440.0, 530.0, 606.0, 760.0, 5000.0, 800.0, 650.0, 650.0]),
        k=np.array([54.0, 48.0, 41.0, 34.0, 29.0, 27.0, 27.0, 27.0]),
        rho=7850.0,
        f_yield=np.array([1.0, 1.0, 0.8, 0.47, 0.2, 0.11, 0.04, 0.0]),
        f_uts=np.array([1.0, 1.0, 1.0, 0.47, 0.2, 0.11, 0.04, 0.0]),
    )


@pytest.fixture
def legacy_steel(steel):
    """The same table as the legacy vfpy Material class."""
    from tests.legacy_ref import legacy
    ht = legacy("heat_transfer")
    return ht.Material(steel.name, steel.T, steel.cp, steel.k, steel.rho, steel.f_yield,
                       steel.f_uts)
