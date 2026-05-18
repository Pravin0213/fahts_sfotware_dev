"""
BELTEMP output file parser — reads USFOS temperature increment results.

BELTEMP format:
    LCASETIM  <load_case>  <time_minutes>
    BELTEMP   <load_case>  <element_id>  <dT_mean>  <dT_grad_y>  <dT_grad_z>

Values are INCREMENTAL temperature changes per output step (not cumulative).
To get absolute temperatures, accumulate: T = T_initial + sum(dT).

Usage:
    records = parse_beltemp("fahts_beltemp.fem")
    temps = cumulative_temperatures(records, T_initial=20.0)
    # temps[eid] = (times_minutes, T_mean_abs) — numpy arrays
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BeltempRecord:
    """One BELTEMP data point: incremental temperatures at a single time step for one element."""
    load_case: int
    time_min: float     # simulation time [minutes]
    element_id: int
    dT_mean: float      # mean temperature increment [°C]
    dT_grad_y: float    # Y-direction gradient increment [°C/m]
    dT_grad_z: float    # Z-direction gradient increment [°C/m]


def parse_beltemp(path: str | Path) -> list[BeltempRecord]:
    """
    Parse a USFOS BELTEMP results file.

    Comments (lines starting with `'`) and blank lines are skipped.
    LCASETIM lines set the current time; subsequent BELTEMP lines inherit it.

    Args:
        path: Path to the BELTEMP .fem file.

    Returns:
        List of BeltempRecord in file order.
    """
    records: list[BeltempRecord] = []
    current_lc: int = 0
    current_time: float = 0.0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("'"):
                continue

            tokens = line.split()
            if not tokens:
                continue

            keyword = tokens[0].upper()

            if keyword == "LCASETIM":
                # LCASETIM  <load_case>  <time_minutes>
                if len(tokens) >= 3:
                    try:
                        current_lc = int(tokens[1])
                        current_time = float(tokens[2])
                    except ValueError:
                        continue

            elif keyword == "BELTEMP":
                # BELTEMP  <load_case>  <element_id>  <dT_mean>  <dT_grad_y>  <dT_grad_z>
                if len(tokens) >= 6:
                    try:
                        lc = int(tokens[1])
                        eid = int(tokens[2])
                        dT_mean = float(tokens[3])
                        dT_grad_y = float(tokens[4])
                        dT_grad_z = float(tokens[5])
                    except ValueError:
                        continue
                    records.append(BeltempRecord(
                        load_case=lc,
                        time_min=current_time,
                        element_id=eid,
                        dT_mean=dT_mean,
                        dT_grad_y=dT_grad_y,
                        dT_grad_z=dT_grad_z,
                    ))

    return records


def cumulative_temperatures(
    records: list[BeltempRecord],
    T_initial: float = 20.0,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert incremental BELTEMP records to absolute temperature histories.

    Accumulates increments starting from T_initial. Returns one entry per
    element ID.

    Args:
        records:   Output of parse_beltemp().
        T_initial: Initial uniform temperature [°C] (default 20.0).

    Returns:
        Dict mapping element_id → (times_min, T_mean_abs, T_grad_y_abs, T_grad_z_abs).
        Each array has shape (n_steps,); times start at the first output step.
        All temperature arrays include the initial state (T_initial, 0, 0) at t=0.
    """
    # Group by element, preserving time order
    by_eid: dict[int, list[BeltempRecord]] = {}
    for r in records:
        by_eid.setdefault(r.element_id, []).append(r)

    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for eid, recs in by_eid.items():
        # Sort by time, then load_case (shouldn't matter if file is ordered)
        recs = sorted(recs, key=lambda r: (r.time_min, r.load_case))
        times = np.array([0.0] + [r.time_min for r in recs])
        dT_m = np.array([r.dT_mean for r in recs])
        dT_y = np.array([r.dT_grad_y for r in recs])
        dT_z = np.array([r.dT_grad_z for r in recs])

        T_mean = np.concatenate([[T_initial], T_initial + np.cumsum(dT_m)])
        T_gy   = np.concatenate([[0.0], np.cumsum(dT_y)])
        T_gz   = np.concatenate([[0.0], np.cumsum(dT_z)])

        result[eid] = (times, T_mean, T_gy, T_gz)

    return result


def element_temperature_at(
    cum_temps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    element_id: int,
    time_min: float,
) -> float:
    """
    Look up the absolute mean temperature [°C] for an element at a given time.

    Uses linear interpolation between stored time steps.

    Args:
        cum_temps:  Output of cumulative_temperatures().
        element_id: Element ID to query.
        time_min:   Query time [minutes].

    Returns:
        Interpolated mean temperature [°C].

    Raises:
        KeyError: If element_id is not in cum_temps.
    """
    times, T_mean, _, _ = cum_temps[element_id]
    return float(np.interp(time_min, times, T_mean))
