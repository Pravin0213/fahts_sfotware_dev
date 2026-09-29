"""fahts.relief must reproduce legacy valves.py / valves2.py / vessel.PSV bit for bit."""

from __future__ import annotations

import pytest

from fahts.relief import blowdown, ideal_gas
from fahts.relief.psv import PSVOpening
from tests.legacy_ref import legacy

pytest.importorskip("CoolProp")

LINE = dict(d=0.1143, t=0.00602, L=20.0)
GASES = {"H2": {"H2": 1.0}, "CH4": {"C1": 1.0},
         "rich": {"C1": 0.8, "C2": 0.1, "C3": 0.06, "C4": 0.04}}


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("line", [None, LINE], ids=["orifice", "line"])
def test_blowdown_mdot_sequence(gas, line):
    """A depressurisation-like sequence, so the warm-start caches are exercised identically."""
    old = legacy("valves2")
    y = GASES[gas]
    for P0, T0 in [(100e5, 293.15), (80e5, 280.0), (60e5, 270.0), (40e5, 262.0), (20e5, 255.0),
                   (5e5, 250.0), (1.2e5, 250.0)]:
        for d in (0.01, 0.04):
            a = blowdown.mdot(P0, T0, y, 101325.0, 0.84, d, line=line)
            b = old.mdot(P0, T0, y, 101325.0, 0.84, d, line=line)
            assert a == b, (P0, T0, d)


def test_blowdown_info_and_nozzles():
    old = legacy("valves2")
    for nozzle in ("api520", "hdi", "hem"):
        a = blowdown.mdot(90e5, 300.0, GASES["rich"], 101325.0, 0.9, 0.02, line=LINE,
                          nozzle=nozzle, info=True)
        b = old.mdot(90e5, 300.0, GASES["rich"], 101325.0, 0.9, 0.02, line=LINE,
                     nozzle=nozzle, info=True)
        assert a == b


def test_ideal_gas_orifice_line():
    old = legacy("valves")
    for P0 in (150e5, 50e5, 3e5):
        args = (P0, 300.0, 101325.0, 0.9, 1.3, 0.016, 0.85, 0.02)
        assert ideal_gas.mdot_orifice_line(*args, line=LINE) == \
            old.mdot_orifice_line(*args, line=LINE)
        assert ideal_gas.mdot_orifice_line(*args) == old.mdot_orifice_line(*args)


@pytest.mark.parametrize("typ", [0, 1, 2])
def test_psv_opening(typ):
    spec = dict(d=0.02, cd=0.9, p_set=110e5, p_full=121e5, p_reseat=100e5, type=typ)
    new, old = PSVOpening(spec), legacy("vessel").PSV(None, spec, "api520")
    for P in (100e5, 109e5, 111e5, 115e5, 125e5, 112e5, 105e5, 99e5, 111e5, 130e5):
        assert new.frac(P) == old.frac(P)
