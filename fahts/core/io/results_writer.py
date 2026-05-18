"""
Phase 2.9 — Export heat BC summary to CSV.

Extended in Phase 4.7 for temperature results: CSV summary, temperature
history CSV, VTK (beam centrelines with temperature cell data), Excel.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from fahts.core.heat.bc.view_factor import ElementHeatBC

if TYPE_CHECKING:
    from fahts.core.model.fem_model import FEMModel
    from fahts.core.results.temperature_field import TemperatureField

log = logging.getLogger(__name__)


# ── Phase 2.9 — BC summary exports ───────────────────────────────────────────

def export_bc_summary_csv(
    bcs: Sequence[ElementHeatBC],
    path: Path | str,
    *,
    append: bool = False,
) -> None:
    """
    Write per-element heat-flux BC summary to a CSV file.

    Columns: element_id, t_s, q_top, q_bot, q_left, q_right, q_total  [W/m²]

    Parameters
    ----------
    bcs    : sequence of ElementHeatBC objects (all at the same time step, or mixed)
    path   : output file path (.csv)
    append : if True, append rows without writing the header again
    """
    path = Path(path)
    mode = "a" if append else "w"
    fieldnames = ["element_id", "t_s", "q_top", "q_bot", "q_left", "q_right", "q_total"]

    with path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for bc in bcs:
            ff = bc.face_fluxes
            writer.writerow({
                "element_id": bc.eid,
                "t_s":        bc.t,
                "q_top":      ff.get("top",   0.0),
                "q_bot":      ff.get("bot",   0.0),
                "q_left":     ff.get("left",  0.0),
                "q_right":    ff.get("right", 0.0),
                "q_total":    bc.total_flux,
            })

    log.info("BC summary written → %s  (%d rows)", path, len(bcs))


def export_bc_summary_multi_time(
    bcs_per_time: dict[float, list[ElementHeatBC]],
    path: Path | str,
) -> None:
    """
    Write BC summaries for multiple time steps to a single CSV.

    Parameters
    ----------
    bcs_per_time : {t [s]: list of ElementHeatBC}
    path         : output file path
    """
    path = Path(path)
    times = sorted(bcs_per_time)
    first = True
    for t in times:
        export_bc_summary_csv(bcs_per_time[t], path, append=not first)
        first = False
    log.info("Multi-time BC summary written → %s  (%d time steps)", path, len(times))


# ── Phase 4.7 — Temperature results exports ───────────────────────────────────

def export_peak_temperature_csv(
    result: TemperatureField,
    path: Path | str,
) -> None:
    """
    Export peak-temperature summary (one row per element) to a CSV file.

    Delegates to ``PostProcessor.to_csv()``.

    Columns: element_id, T_initial_C, T_peak_centroid_C, T_peak_nodal_C,
             t_crit_500_s, t_crit_500_min, t_crit_600_s, t_crit_600_min,
             fire_resistance_min, passes_600
    """
    from fahts.core.results.post_processor import PostProcessor

    pp = PostProcessor(result)
    pp.to_csv(Path(path))
    log.info("Peak-temperature CSV written → %s", path)


def export_temperature_history_csv(
    result: TemperatureField,
    path: Path | str,
) -> None:
    """
    Export full centroid temperature time-history to a CSV file.

    Delegates to ``PostProcessor.to_temperature_csv()``.

    Columns: time_s, time_min, T_eid_<id1>, T_eid_<id2>, …
    """
    from fahts.core.results.post_processor import PostProcessor

    pp = PostProcessor(result)
    pp.to_temperature_csv(Path(path))
    log.info("Temperature history CSV written → %s", path)


def export_results_vtk(
    result: TemperatureField,
    model: FEMModel,
    path: Path | str,
) -> None:
    """
    Export temperature results as a VTK file (beam centrelines with cell data).

    The file can be opened in ParaView.  Each beam element in *result*
    appears as a line cell.  Cell data arrays:

    * ``element_id``        — beam element ID (int32)
    * ``T_peak_C``          — peak centroid temperature [°C]
    * ``T_final_C``         — centroid temperature at last time step [°C]
    * ``fire_resistance_min`` — minutes until 600 °C reached (NaN if not)
    * ``passes_600``        — 1 if element never reached 600 °C, else 0

    Field data ``times_s`` stores the simulation time array [s].
    """
    import numpy as np
    import pyvista as pv
    from fahts.core.results.post_processor import PostProcessor

    path = Path(path)
    pp = PostProcessor(result)
    summaries = {s.eid: s for s in pp.element_summaries()}
    centroid = model.centroid()

    pts_list: list[np.ndarray] = []
    lines: list[int] = []
    eids_found: list[int] = []
    pt_idx = 0

    for eid in result.element_ids:
        elem = model.elements.get(eid)
        if elem is None:
            log.warning("VTK export: element %d not in model — skipped", eid)
            continue
        n1, n2 = elem.n1, elem.n2
        if n1 not in model.nodes or n2 not in model.nodes:
            log.warning("VTK export: element %d nodes not found — skipped", eid)
            continue

        p1 = model.nodes[n1].xyz.copy()
        p2 = model.nodes[n2].xyz.copy()
        if elem.ecc1 is not None:
            p1 += elem.ecc1
        if elem.ecc2 is not None:
            p2 += elem.ecc2
        pts_list.append(p1 - centroid)
        pts_list.append(p2 - centroid)
        lines += [2, pt_idx, pt_idx + 1]
        pt_idx += 2
        eids_found.append(eid)

    if not eids_found:
        log.warning("VTK export: no matching elements found — file not written.")
        return

    import numpy as np  # already imported above, just for clarity
    pts = np.array(pts_list, dtype=float)
    mesh = pv.PolyData(pts, lines=np.array(lines, dtype=np.intp))

    # ── Cell data ─────────────────────────────────────────────────────────────
    mesh.cell_data["element_id"] = np.array(eids_found, dtype=np.int32)
    mesh.cell_data["T_peak_C"] = np.array(
        [result.peak_centroid_temperature(eid) for eid in eids_found], dtype=float
    )
    mesh.cell_data["T_final_C"] = np.array(
        [result.centroid_temperature(eid, t_idx=-1) for eid in eids_found], dtype=float
    )

    fr_arr = []
    passes_arr = []
    for eid in eids_found:
        s = summaries[eid]
        fr_arr.append(s.fire_resistance_min if s.fire_resistance_min is not None else float("nan"))
        passes_arr.append(1 if s.passes_600 else 0)
    mesh.cell_data["fire_resistance_min"] = np.array(fr_arr, dtype=float)
    mesh.cell_data["passes_600"] = np.array(passes_arr, dtype=np.int8)

    # ── Per-step arrays (T at each output time) ───────────────────────────────
    eid_col = {eid: result._eid_to_idx[eid] for eid in eids_found}
    for step_i, t_val in enumerate(result.times):
        col_data = np.array(
            [result.T_centroid[step_i, eid_col[eid]] for eid in eids_found], dtype=float
        )
        mesh.cell_data[f"T_step{step_i:03d}_C"] = col_data

    # ── Field data (simulation times for reference) ───────────────────────────
    mesh.field_data["times_s"] = result.times.copy()

    mesh.save(str(path))
    n_steps = len(result.times)
    log.info(
        "VTK written → %s  (%d elements, %d time steps)",
        path, len(eids_found), n_steps,
    )


def export_results_excel(
    result: TemperatureField,
    path: Path | str,
) -> None:
    """
    Export temperature results to an Excel workbook (.xlsx).

    Sheets
    ------
    Summary
        Per-element peak temperature, fire-resistance rating, PASS/FAIL
        (one row per element).
    T_History
        Full centroid temperature time-history for every element
        (rows = time steps, columns = elements).

    Requires ``pandas`` and ``openpyxl`` to be installed.
    """
    import pandas as pd  # noqa: PLC0415
    from fahts.core.results.post_processor import PostProcessor  # noqa: PLC0415

    path = Path(path)
    pp = PostProcessor(result)

    df_summary = pp.to_dataframe().reset_index()
    df_history = pp.to_history_dataframe().reset_index()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_history.to_excel(writer, sheet_name="T_History", index=False)

    log.info(
        "Excel written → %s  (%d elements, %d steps)",
        path, result.n_elements, result.n_steps,
    )


def export_beltemp(
    result: TemperatureField,
    path: Path | str,
    *,
    T_initial: float = 20.0,
    time_unit: str = "s",
    meshes: dict | None = None,
) -> None:
    """
    Export temperature results in USFOS BELTEMP format.

    Each output time step produces a LCASETIM header followed by BELTEMP records
    (one per element).  Values are INCREMENTAL temperature changes (matching the
    USFOS convention).

    When *meshes* is supplied (mapping eid → SectionMesh) the Y/Z gradient columns
    are computed from the linearised first-moment formula (SINTEF FAHTS §3.4.2):

        βz = Σ(T_k · y_k · A_k) / Iz
        βy = Σ(T_k · z_k · A_k) / Iy

    Elements without a corresponding mesh entry receive zero gradients.

    Args:
        result:     TemperatureField from run_analysis().
        path:       Output file path (typically .fem).
        T_initial:  Initial temperature used to compute increments (default 20.0 °C).
        time_unit:  "s" (seconds, default) or "min" (minutes) for displayed time values.
        meshes:     Optional dict mapping element IDs to SectionMesh objects.
                    When provided, actual βy/βz gradients are written instead of zeros.
    """
    path = Path(path)
    n_steps = result.n_steps
    n_elems = result.n_elements
    _meshes = meshes or {}

    def _time_val(t_s: float) -> float:
        return t_s / 60.0 if time_unit == "min" else t_s

    with path.open("w", encoding="utf-8") as fh:
        fh.write("'\n")
        fh.write("'  Temperature increments exported by FAHTS\n")
        fh.write("'\n")
        fh.write(f"'  TimeUnit: {time_unit}\n")
        fh.write("'\n")

        for step_i in range(n_steps):
            lc = step_i + 1
            t_disp = _time_val(float(result.times[step_i]))
            fh.write(f"\n LCASETIM{lc:11d}{t_disp:14.4f}\n")
            fh.write(
                "'              load case   element     Mean       "
                "Gradient    Gradient\n"
            )
            fh.write(
                "'                  no      number   temperature     "
                "Y-dir       Z-dir\n"
            )
            fh.write(
                "'                                   (increment)  "
                "(increment) (increment)\n"
            )

            for col, eid in enumerate(result.element_ids):
                T_now = float(result.T_centroid[step_i, col])
                T_prev = (
                    float(result.T_centroid[step_i - 1, col])
                    if step_i > 0
                    else T_initial
                )
                dT = T_now - T_prev

                if eid in _meshes:
                    mesh = _meshes[eid]
                    by_now, bz_now = result.section_gradient(eid, step_i, mesh)
                    by_prev, bz_prev = (
                        result.section_gradient(eid, step_i - 1, mesh)
                        if step_i > 0
                        else (0.0, 0.0)
                    )
                    d_beta_y = by_now - by_prev
                    d_beta_z = bz_now - bz_prev
                else:
                    d_beta_y = 0.0
                    d_beta_z = 0.0

                fh.write(
                    f" BELTEMP{lc:12d}{eid:12d}{dT:14.3f}"
                    f"{d_beta_y:12.3f}{d_beta_z:12.3f}\n"
                )

    log.info(
        "BELTEMP written → %s  (%d elements, %d steps)",
        path, n_elems, n_steps,
    )
