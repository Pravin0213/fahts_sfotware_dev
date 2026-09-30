"""3-D vessel geometry: region areas agree with the model's wall-region fractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista")

from fahts.coupling import VesselFireModel  # noqa: E402
from fahts.coupling.vessfire_import import case_from_vessfire_deck  # noqa: E402
from fahts.renderer.vessel_geometry import (VesselGeometry3D, region_temperatures,  # noqa: E402
                                            wetted_half_angle_deg)

CASES = Path(__file__).resolve().parents[2] / "regression" / "process" / "cases"


@pytest.mark.parametrize("case_id", ["M06-0030", "M06-0069", "M06-0070", "M14-0095", "M11-0002"])
def test_region_areas_match_model_fractions(case_id, steel):
    case, _ = case_from_vessfire_deck(CASES / case_id)
    model = VesselFireModel(case.to_model_case(), case.model_options(), steel)
    g = VesselGeometry3D(case, n_theta=720, n_length=200, peak_modelled=model.peak is not None)
    g.set_level(model._level())
    got = g.region_fractions()
    for k, f in model.frac.items():
        assert got[k] == pytest.approx(f, abs=3e-3), (k, got, model.frac)


def test_wetted_angle_and_liquid_body():
    assert wetted_half_angle_deg(0.0, 2.0) == 0.0
    assert wetted_half_angle_deg(1.0, 2.0) == pytest.approx(90.0)
    assert wetted_half_angle_deg(2.0, 2.0) == 180.0
    case, _ = case_from_vessfire_deck(CASES / "M11-0002")          # 1 m liquid in D = 2 m
    g = VesselGeometry3D(case)
    g.set_level(1.0)
    body = g.liquid_body()
    assert body.volume == pytest.approx(np.pi / 2 * 1.0**2 * g.L, rel=2e-3)   # half-full
    g.set_level(0.0)
    assert g.liquid_body() is None


def test_paint_from_results():
    import pandas as pd
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")
    g = VesselGeometry3D(case)
    g.set_level(1.0)
    ts = pd.DataFrame({"background_T_mean_C": [100.0], "wet_T_mean_C": [50.0],
                       "peak_T_mean_C": [700.0], "peak_wet_T_mean_C": [900.0]})
    g.paint(region_temperatures(ts, 0))
    T, r = g.shell.cell_data["T_C"], g.shell.cell_data["region"]
    assert set(T[r == 2]) == {700.0} and set(T[r == 1]) == {50.0}
    assert not (r == 3).any()                    # jet on top: no wetted peak cells


def _jet_case():
    case, _ = case_from_vessfire_deck(CASES / "M06-0070")          # D 2 m, t 60 mm, jet on top
    return case


@pytest.mark.parametrize("scale", [1.0, 5.0])
def test_solid_wall_volume_and_cutaways(scale):
    case = _jet_case()
    g = VesselGeometry3D(case, thickness_scale=scale)
    g.set_level(1.0)
    surf = g.wall_surface()
    R, t, L = g.R, case.vessel.wall_m * scale, g.L
    assert g.wall.volume == pytest.approx(np.pi * ((R + t) ** 2 - R ** 2) * L, rel=2e-3)
    n = g.wall.n_cells
    half = g.wall.extract_cells(np.flatnonzero(~((g._w_theta > 0) & (g._w_theta < 180))))
    assert half.n_cells == pytest.approx(n / 2, rel=0.02)
    assert g.wall_surface("quarter").n_cells > 0 and surf.n_cells > 0
    assert {0, 1, 2} <= set(np.unique(surf.cell_data["region"]))   # dry, wet, jet (dry)


@pytest.mark.parametrize("scale", [1.0, 5.0])
def test_through_thickness_painting(scale):
    case = _jet_case()
    g = VesselGeometry3D(case, thickness_scale=scale)
    g.set_level(1.0)
    x = np.linspace(0.0, case.vessel.wall_m, 12)
    T_nodes = 100.0 + 500.0 * x / x[-1]                             # 100 C inside, 600 C outside
    g.paint_wall({"dry": (x, T_nodes), "wet": 50.0, "peak_dry": (x, T_nodes + 100.0)})
    T, depth, region = g.wall.cell_data["T_C"], g._w_depth, g.wall.cell_data["region"]
    dry = region == 0
    np.testing.assert_allclose(T[dry], 100.0 + 500.0 * depth[dry] / case.vessel.wall_m,
                               rtol=1e-9)
    assert T[dry].min() < 140.0 and T[dry].max() > 560.0            # gradient through the wall
    assert set(T[region == 1]) == {50.0}
