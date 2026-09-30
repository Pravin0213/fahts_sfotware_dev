"""Wall node layout follows the actual wall thickness."""

from __future__ import annotations

import numpy as np
import pytest

from fahts.wall.column_1d import radial_nodes


@pytest.mark.parametrize("t", [0.006, 0.06, 0.1, 0.105])
def test_layout_spans_the_wall(t):
    x = radial_nodes(t)
    assert len(x) == 12 and x[0] == 0.0 and x[-1] == pytest.approx(t, rel=1e-15)
    np.testing.assert_allclose(np.diff(x[1:-1]), t / 10)          # equal interior cells
    np.testing.assert_allclose([x[1], t - x[-2]], [t / 20, t / 20])  # half cells at surfaces
    assert np.all(np.diff(x) > 0)


def test_cells_and_errors():
    assert len(radial_nodes(0.05, n_cells=20)) == 22
    with pytest.raises(ValueError):
        radial_nodes(0.0)
    with pytest.raises(ValueError):
        radial_nodes(0.05, n_cells=0)
