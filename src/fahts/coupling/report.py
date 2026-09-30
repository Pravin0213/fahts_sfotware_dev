"""Plain-text summary of a vessel-in-fire result (shared by the GUI and the command line)."""

from __future__ import annotations

import math

from fahts.coupling.runner import CaseResult

REGIONS = (("background", "dry wall"), ("wet", "wetted wall"), ("peak", "peak zone (dry)"),
           ("peak_wet", "peak zone (wetted)"))


def _fmt_time(t) -> str:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "—"
    return f"{t:.0f} s ({t / 60:.1f} min)"


def regions_with_area(res: CaseResult) -> list[str]:
    """Wall regions that have area at some time. Regions without area are still integrated
    by the model but their temperatures are meaningless, so they are not shown."""
    ts = res.series
    out = []
    for col, _ in REGIONS:
        if f"{col}_T_mean_C" not in ts:
            continue
        if col == "wet":
            has = ts.f_wet.max() > 0
        elif col == "peak_wet":
            has = ts.f_peak_wet.max() > 0
        elif col == "peak":
            p = res.case.heat_load.peak
            f_peak = max(min(p.xi_end, 1.0) - max(p.xi_start, 0.0), 0.0) * p.circ_deg / 360.0
            has = (f_peak - ts.f_peak_wet).max() > 1e-12
        else:
            has = True
        if has:
            out.append(col)
    return out


def summary_lines(res: CaseResult) -> list[tuple[str, str]]:
    """(label, value) pairs describing a result (also used by tests and reports)."""
    ts, fail = res.series, res.failures
    first = fail.dropna(subset=["t_fail_s"]).sort_values("t_fail_s")
    earliest = ("—" if first.empty else
                f"{_fmt_time(first.t_fail_s.iloc[0])} — {first.variant.iloc[0]}, "
                f"{first.criterion.iloc[0]}, {first.allow_basis.iloc[0]}")
    walls = [label for c, label in REGIONS if c in regions_with_area(res)]
    return [
        ("Case", res.case.name),
        ("Rupture, membrane stress (Tresca)", _fmt_time(res.rupture_tresca_s)),
        ("Rupture, membrane stress (von Mises)", _fmt_time(res.rupture_von_mises_s)),
        ("Earliest failure, any stress check (incl. thermal stress)", earliest),
        ("Peak pressure", f"{ts.P_bara.max():.2f} bara at {ts.Time[ts.P_bara.idxmax()]:.0f} s"),
        ("Hottest wall (through-wall mean)", f"{ts.T_mean_hot_C.max():.0f} °C"),
        ("Gas temperature, max", f"{ts.T_gas_C.max():.0f} °C"),
        ("Released mass", f"{ts.released.iloc[-1]:,.1f} kg"),
        ("Wall regions", ", ".join(walls)),
        ("Energy balance error", f"{ts.energy_err_pct.iloc[-1]:.3f} %"),
        ("Run time", f"{res.runtime_s:.1f} s"),
    ]
