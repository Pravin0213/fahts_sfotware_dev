"""3-D vessel geometry: region areas agree with the model's wall-region fractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista")

from fahts.coupling import VesselFireModel  # noqa: E402
from fahts.coupling.vessfire_import import case_from_vessfire_deck  # noqa: E402
from fahts.renderer.vessel_geometry import (VesselGeometry3D,  # noqa: E402
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


def test_seam_is_closed_where_the_jet_crosses_the_top():
    """A zone across 0/360 deg must be one patch: no outline edge along the seam."""
    case = _jet_case()                                    # jet centred on the top (0 deg)
    g = VesselGeometry3D(case)
    zone = g.shell.extract_cells(np.flatnonzero(g.shell.cell_data["region"] >= 2))
    edges = zone.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                       manifold_edges=False, non_manifold_edges=False)
    th = np.degrees(np.arctan2(edges.points[:, 1], edges.points[:, 2])) % 360.0
    on_seam = (th < 1e-6) | (th > 360 - 1e-6)
    seg = edges.lines.reshape(-1, 3)[:, 1:]                       # (n_edges, 2) point ids
    assert not np.any(on_seam[seg[:, 0]] & on_seam[seg[:, 1]])      # no edge ALONG the seam
    assert g.shell.n_points == (len(g._th) - 1) * len(g._xs)             # seam merged
    assert np.diff(g._xs).min() > 1e-6 and np.diff(g._th).min() > 1e-6    # no slivers
    surf = g.wall_surface()
    x = surf.cell_centers().points
    th_c = np.degrees(np.arctan2(x[:, 1], x[:, 2])) % 360.0
    r_c = np.hypot(x[:, 1], x[:, 2])
    inside_wall = (r_c > g.R + 1e-4) & (r_c < g.R_out - 1e-4)
    assert not np.any(inside_wall & ((th_c < 1e-3) | (th_c > 360 - 1e-3)))  # no seam faces
