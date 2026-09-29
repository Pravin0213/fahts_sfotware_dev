"""Radial node layouts through the wall."""

from __future__ import annotations

import numpy as np

# Node positions through the wall, measured from the INNER surface [m], for a 105 mm shell:
# two surface nodes plus ten interior nodes (half cell at each surface: 0, t/20, 3t/20, ...,
# then a thin outer cell). Taken from the calibration reference's run log.
#
# KNOWN ISSUE (docs/process_model_known_issues.md #1): the process model uses this layout
# for every vessel, whatever its wall thickness. Kept unchanged until the port is complete.
REFERENCE_NODES_105MM = np.array([
    0.0, 0.00525, 0.01575, 0.02625, 0.03675, 0.04725,
    0.05775, 0.06825, 0.07875, 0.08925, 0.1017, 0.105])
