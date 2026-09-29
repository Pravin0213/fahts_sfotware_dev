"""fahts.process geometry + inner_ht must reproduce legacy twophase_physics bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.process.geometry import VesselGeometry, layers
from fahts.process.inner_ht import boiling_curve, condensation, interface, natural_convection
from fahts.process.inner_ht.boiling_correlations import nb_cooper, q_chf
from fahts.process.inner_ht.natural_convection import H_CORRELATIONS
from tests.legacy_ref import legacy


@pytest.fixture(scope="module")
def tp():
    return legacy("twophase_physics")


GEOMS = [dict(D=2.0, L=6.0, orientation="horizontal", head="flat"),
         dict(D=2.58, L=24.65, orientation="horizontal", head="hemispherical"),
         dict(D=1.5, L=4.0, orientation="vertical", head="ellipsoidal")]

SAT = dict(T_sat=320.0, rho_l=480.0, rho_v=40.0, h_l=2.0e5, h_v=5.5e5, h_fg=3.5e5,
           sigma=0.006, mu_l=1.0e-4, mu_v=1.0e-5, k_l=0.09, k_v=0.02, cp_l=2800.0,
           cp_v=2000.0, P=15e5, P_c=42.5e5, M=44.1)
LIQ = dict(rho=480.0, cp=2800.0, mu=1.0e-4, k=0.09, beta=3e-3)
GAS = dict(rho=40.0, cp=2000.0, mu=1.0e-5, k=0.02, beta=3.2e-3)


@pytest.mark.parametrize("g", GEOMS, ids=["h-flat", "h-hemi", "v-ellip"])
def test_geometry(tp, g):
    new, old = VesselGeometry(**g), tp.VesselGeometry(**g)
    assert new.V_total == old.V_total
    for f in np.linspace(0.02, 0.98, 9):
        h = new.level(f * new.V_total)
        assert h == old.level(f * old.V_total)
        for attr in ("volume", "interface_area", "wetted_area", "wetted_fraction",
                     "wetted_perimeter_fraction", "interface_char_length",
                     "liquid_char_length", "gas_char_length"):
            assert getattr(new, attr)(h) == getattr(old, attr)(h), (attr, f)
    ln, lo = layers(new, 0.1 * new.V_total, 0.3 * new.V_total), \
        tp.layers(old, 0.1 * old.V_total, 0.3 * old.V_total)
    assert ln.__dict__ == lo.__dict__


def test_natural_convection(tp):
    for dT in (-30.0, 0.5, 20.0, 300.0):
        for cfg in ("vertical_plate", "horizontal_cylinder", "hot_plate_up", "cold_plate_down"):
            try:
                a = natural_convection.h_free(GAS, dT, 2.0, cfg)
            except (KeyError, ValueError) as e:
                with pytest.raises(type(e)):
                    tp.h_free(GAS, dT, 2.0, cfg)
                continue
            assert a == tp.h_free(GAS, dT, 2.0, cfg)
    old = legacy("vessel").H_CORRELATIONS
    for k, f in H_CORRELATIONS.items():
        assert f(3.7e9, 0.8) == old[k](3.7e9, 0.8)


@pytest.mark.parametrize("wall_over_sat", [-5.0, 2.0, 15.0, 60.0, 400.0])
def test_boiling_curve(tp, wall_over_sat):
    T_w = SAT["T_sat"] + wall_over_sat
    for model in (boiling_curve.BoilingModel(), boiling_curve.BoilingModel(nucleate="rohsenow")):
        old_model = tp.BoilingModel(**model.__dict__)
        a = boiling_curve.boiling_flux_and_derivative(T_w, 300.0, SAT, LIQ, 1.0, model)
        b = tp.boiling_flux_and_derivative(T_w, 300.0, SAT, LIQ, 1.0, old_model)
        assert a == b
    assert nb_cooper(SAT)(12.0) == tp.nb_cooper(SAT)(12.0)
    assert q_chf(SAT) == tp.q_chf(SAT)
    assert boiling_curve.linearised_bc(5e4, 800.0, 400.0) == tp.linearised_bc(5e4, 800.0, 400.0)


def test_condensation_and_interface(tp):
    for T_w in (280.0, 310.0):
        assert condensation.condensation_flux(T_w, SAT) == tp.condensation_flux(T_w, SAT)
    a = interface.interface_exchange(340.0, 310.0, SAT, GAS, LIQ, 12.0, 2.0)
    b = tp.interface_exchange(340.0, 310.0, SAT, GAS, LIQ, 12.0, 2.0)
    assert a == b
