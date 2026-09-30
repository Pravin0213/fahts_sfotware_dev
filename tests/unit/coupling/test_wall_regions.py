"""Area split of the shell into background / peak fire zones x dry / wet contact."""

from __future__ import annotations

import pytest

from fahts.coupling.wall_regions import PeakZone, _arc_overlap_deg, region_fractions


def test_no_peak_is_the_plain_dry_wet_split():
    assert region_fractions(0.3, None) == {"dry": 0.7, "wet": 0.3, "peak_dry": 0.0,
                                           "peak_wet": 0.0}


@pytest.mark.parametrize("f_wet", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("attack", [0.0, 90.0, 180.0, 270.0, 350.0])
def test_fractions_sum_to_one_and_are_non_negative(f_wet, attack):
    f = region_fractions(f_wet, PeakZone(0.4, 0.6, 90.0, attack))
    assert sum(f.values()) == pytest.approx(1.0)
    assert min(f.values()) >= 0.0
    assert f["peak_dry"] + f["peak_wet"] == pytest.approx(0.2 * 90 / 360)
    assert f["wet"] + f["peak_wet"] == pytest.approx(f_wet)


def test_peak_position_relative_to_liquid():
    p_bottom = PeakZone(0.4, 0.6, 90.0, 180.0)      # centred on the bottom
    p_top = PeakZone(0.4, 0.6, 90.0, 0.0)
    f_half = 0.5                                     # liquid to mid-height: wetted arc 90..270
    assert region_fractions(f_half, p_bottom)["peak_dry"] == pytest.approx(0.0)
    assert region_fractions(f_half, p_top)["peak_wet"] == pytest.approx(0.0)
    # side attack (90 deg) with half fill: arc 45..135, half of it below 90 deg from bottom
    side = region_fractions(f_half, PeakZone(0.4, 0.6, 90.0, 90.0))
    assert side["peak_wet"] == pytest.approx(side["peak_dry"])


def test_arc_overlap_wraps_around_the_top():
    assert _arc_overlap_deg(350.0, 40.0, 10.0, 40.0) == pytest.approx(20.0)
    assert _arc_overlap_deg(0.0, 90.0, 180.0, 90.0) == pytest.approx(0.0)
    assert _arc_overlap_deg(180.0, 360.0, 0.0, 30.0) == pytest.approx(30.0)
