"""fahts.thermo must reproduce the legacy vfpy PR model bit for bit (pure port, no changes)."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.thermo import PRMixture
from tests.legacy_ref import legacy

pytest.importorskip("CoolProp")

FLUIDS = {
    "hydrogen": ({"H2": 1.0}, None),
    "methane": ({"C1": 1.0}, None),
    "lpg": ({"C3": 0.6, "C4": 0.4}, None),
    "rich_gas": ({"C1": 0.75, "C2": 0.08, "C3": 0.06, "C4": 0.04, "C6": 0.03, "CO2": 0.02,
                  "N2": 0.02}, None),
    "pseudo": ({"C1": 0.7, "PSEU1": 0.3}, {"PSEU1": {"sg": 0.7, "Tb": 350.0}}),
    "gas_water": ({"C1": 0.8, "C3": 0.1, "H2O": 0.1}, None),
}
STATES = [(1e5, 250.0), (20e5, 293.15), (50e5, 320.0), (100e5, 400.0), (150e5, 600.0)]


def _pair(fluid):
    comp, pseudo = FLUIDS[fluid]
    return PRMixture(comp, pseudo=pseudo), legacy("thermo_pr").PRMixture(comp, pseudo=pseudo)


def _outcome(fn, *args):
    """Value of fn(*args), or the exception type and message (both models must agree)."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - comparing failure behaviour on purpose
        return (type(e).__name__, str(e))


def _same_result(a, b):
    assert a.phase_names == b.phase_names
    for x, y in ((a.T, b.T), (a.P, b.P)):
        assert x == y
    for pa, pb in zip(a.phases, b.phases):
        for attr in ("beta", "Z", "v", "M", "h", "u", "s", "cp", "cv", "w"):
            assert getattr(pa, attr) == getattr(pb, attr), attr
        np.testing.assert_array_equal(pa.x, pb.x)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_flash_pt(fluid):
    new, old = _pair(fluid)
    for P, T in STATES:
        _same_result(new.flash_PT(P, T), old.flash_PT(P, T))


@pytest.mark.parametrize("fluid", FLUIDS)
def test_flash_ph_and_uv(fluid):
    new, old = _pair(fluid)
    for P, T in STATES[1:4]:
        r_new, r_old = new.flash_PT(P, T), old.flash_PT(P, T)
        _same_result(new.flash_PH(P * 0.8, r_new.h), old.flash_PH(P * 0.8, r_old.h))
        n = new.z * 1000.0
        U, V = r_new.u * 1000.0, r_new.v * 1000.0 * 1.05
        _same_result(new.flash_UV(U, V, n, T_guess=T, init=r_new),
                     old.flash_UV(U, V, n, T_guess=T, init=r_old))


@pytest.mark.parametrize("fluid", ["lpg", "rich_gas", "pseudo"])
def test_saturation_and_transport(fluid):
    new, old = _pair(fluid)
    for method, arg in (("bubble_P", 280.0), ("dew_P", 280.0), ("bubble_T", 10e5),
                        ("dew_T", 10e5)):
        a = _outcome(getattr(new, method), arg)
        b = _outcome(getattr(old, method), arg)
        assert type(a) is type(b)
        if isinstance(a, tuple) and isinstance(a[0], str):
            assert a == b
        else:
            assert a[0] == b[0]
            np.testing.assert_array_equal(a[1], b[1])
    r_new, r_old = new.transport(new.flash_PT(10e5, 280.0)), old.transport(old.flash_PT(10e5, 280.0))
    for pa, pb in zip(r_new.phases, r_old.phases):
        assert (pa.mu, pa.k) == (pb.mu, pb.k)
    assert r_new.sigma == r_old.sigma
