"""
Task 3.9 — PostProcessor: derived quantities and tabular exports.

Wraps a TemperatureField and provides:
  - Per-element summary (peak, time-to-critical, pass/fail)
  - Peak nodal temperature (hotspot, when section data is available)
  - Fire-resistance rating in minutes
  - Full centroid time-history as a wide table
  - CSV and pandas DataFrame export

All methods are pure (no side effects).  Export methods write to disk.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fahts.core.results.temperature_field import TemperatureField

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

# EN 1993-1-2 Table 4.1 — standard critical temperature for load ratio μ₀ = 0.5
T_CRIT_DEFAULT: float = 600.0   # °C


@dataclass(frozen=True)
class ElementSummary:
    """
    Condensed fire-performance summary for one beam element.

    Attributes
    ----------
    eid:
        Beam element ID.
    T_initial:
        Centroid temperature at t = 0 [°C].
    T_peak_centroid:
        Maximum area-weighted centroid temperature over the full simulation [°C].
    T_peak_nodal:
        Maximum temperature of any cross-section node over the full simulation [°C].
        ``None`` if section nodal data was not stored.
    t_crit_500:
        Time [s] when centroid first reaches 500 °C, or ``None``.
    t_crit_600:
        Time [s] when centroid first reaches 600 °C, or ``None``.
        This is the standard EN 1993-1-2 fire-resistance criterion.
    fire_resistance_min:
        ``t_crit_600`` expressed in minutes, or ``None`` if never reached.
    passes_600:
        True when the element never reached 600 °C during the simulation.
    """

    eid: int
    T_initial: float
    T_peak_centroid: float
    T_peak_nodal: float | None
    t_crit_500: float | None
    t_crit_600: float | None

    @property
    def fire_resistance_min(self) -> float | None:
        """t_crit_600 in minutes, or None."""
        return self.t_crit_600 / 60.0 if self.t_crit_600 is not None else None

    @property
    def passes_600(self) -> bool:
        """Element never exceeded 600 °C → passes standard fire-resistance check."""
        return self.t_crit_600 is None


class PostProcessor:
    """
    Derives fire-performance quantities from a completed TemperatureField.

    Parameters
    ----------
    result:
        TemperatureField produced by ``run_analysis`` or ``TemperatureField.merge``.

    Example
    -------
    ::

        pp = PostProcessor(temperature_field)
        summaries = pp.element_summaries()
        pp.to_csv("results_summary.csv")
        df = pp.to_dataframe()
    """

    def __init__(self, result: TemperatureField) -> None:
        self._tf = result

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def result(self) -> TemperatureField:
        """The underlying TemperatureField."""
        return self._tf

    @property
    def n_elements(self) -> int:
        return self._tf.n_elements

    @property
    def n_steps(self) -> int:
        return self._tf.n_steps

    @property
    def duration_s(self) -> float:
        """Total simulated fire duration [s]."""
        return float(self._tf.times[-1])

    @property
    def duration_min(self) -> float:
        """Total simulated fire duration [minutes]."""
        return self.duration_s / 60.0

    # ── Per-element summary ───────────────────────────────────────────────────

    def element_summary(self, eid: int) -> ElementSummary:
        """
        Build an ElementSummary for one beam element.

        Parameters
        ----------
        eid:
            Element ID.

        Returns
        -------
        ElementSummary
        """
        tf = self._tf
        T_init = tf.centroid_temperature(eid, t_idx=0)
        T_peak_cen = tf.peak_centroid_temperature(eid)

        T_peak_nodal: float | None = None
        if eid in tf.T_section:
            T_peak_nodal = float(tf.T_section[eid].max())

        t500 = tf.time_to_critical(eid, T_crit=500.0)
        t600 = tf.time_to_critical(eid, T_crit=600.0)

        return ElementSummary(
            eid=eid,
            T_initial=T_init,
            T_peak_centroid=T_peak_cen,
            T_peak_nodal=T_peak_nodal,
            t_crit_500=t500,
            t_crit_600=t600,
        )

    def element_summaries(self) -> list[ElementSummary]:
        """
        Return ElementSummary for every element, ordered as in the TemperatureField.

        Returns
        -------
        list[ElementSummary]
        """
        return [self.element_summary(eid) for eid in self._tf.element_ids]

    # ── Bulk queries ──────────────────────────────────────────────────────────

    def fire_resistance_minutes(
        self, eid: int, T_crit: float = T_CRIT_DEFAULT
    ) -> float | None:
        """
        Fire resistance rating for element ``eid`` [minutes].

        Returns None if the critical temperature was not reached.
        """
        t_s = self._tf.time_to_critical(eid, T_crit=T_crit)
        return t_s / 60.0 if t_s is not None else None

    def critical_elements(self, T_crit: float = T_CRIT_DEFAULT) -> list[int]:
        """
        Element IDs whose peak centroid temperature reached or exceeded ``T_crit``.
        Preserves element_ids order.
        """
        return self._tf.critical_elements(T_crit)

    def passing_elements(self, T_crit: float = T_CRIT_DEFAULT) -> list[int]:
        """
        Element IDs whose peak centroid temperature stayed below ``T_crit``.
        Preserves element_ids order.
        """
        crit_set = set(self.critical_elements(T_crit))
        return [eid for eid in self._tf.element_ids if eid not in crit_set]

    def peak_temperatures(self) -> dict[int, float]:
        """Peak centroid temperature for every element: {eid: T_peak [°C]}."""
        return self._tf.peak_temperatures

    def minimum_fire_resistance(
        self, T_crit: float = T_CRIT_DEFAULT
    ) -> float | None:
        """
        Minimum fire resistance [minutes] across all elements that reached T_crit.
        Returns None if no element reached T_crit.
        """
        times = [
            self.fire_resistance_minutes(eid, T_crit)
            for eid in self._tf.element_ids
        ]
        valid = [t for t in times if t is not None]
        return min(valid) if valid else None

    # ── Time-history tables ───────────────────────────────────────────────────

    def centroid_history(self, eid: int) -> dict[str, np.ndarray]:
        """
        Centroid temperature history for one element.

        Returns
        -------
        dict with keys ``'times_s'``, ``'times_min'``, and ``'T_centroid_C'``;
        all arrays of length ``n_steps``.
        """
        idx = self._tf._eid_index(eid)
        T = self._tf.T_centroid[:, idx]
        return {
            "times_s":     self._tf.times.copy(),
            "times_min":   self._tf.times / 60.0,
            "T_centroid_C": T.copy(),
        }

    def all_centroid_history(self) -> dict[str, np.ndarray]:
        """
        Wide centroid temperature table for all elements.

        Returns
        -------
        dict with keys ``'times_s'``, ``'times_min'``, and one key per element ID
        (``'T_eid_{eid}'``), each containing an (n_steps,) array [°C].
        """
        data: dict[str, np.ndarray] = {
            "times_s":   self._tf.times.copy(),
            "times_min": self._tf.times / 60.0,
        }
        for i, eid in enumerate(self._tf.element_ids):
            data[f"T_eid_{eid}"] = self._tf.T_centroid[:, i].copy()
        return data

    # ── Text summary ──────────────────────────────────────────────────────────

    def summary_text(self, T_crit: float = T_CRIT_DEFAULT) -> str:
        """
        Multi-line human-readable summary of the analysis results.

        Parameters
        ----------
        T_crit:
            Critical temperature threshold [°C] for pass/fail classification.
        """
        summaries = self.element_summaries()
        n_crit = sum(1 for s in summaries if not s.passes_600)
        lines = [
            f"Heat Transfer Analysis Results",
            f"  Duration:        {self.duration_min:.1f} min",
            f"  Elements solved: {self.n_elements}",
            f"  Output steps:    {self.n_steps}",
            f"  Critical (≥ {T_crit:.0f} °C): {n_crit} / {self.n_elements}",
            "",
            f"{'EID':>6}  {'T_init':>8}  {'T_peak':>8}  "
            f"{'t_crit500':>10}  {'t_crit600':>10}  {'FR [min]':>9}  Status",
            "-" * 75,
        ]
        for s in summaries:
            t500 = f"{s.t_crit_500/60:.1f} min" if s.t_crit_500 is not None else "   —"
            t600 = f"{s.t_crit_600/60:.1f} min" if s.t_crit_600 is not None else "   —"
            fr   = f"{s.fire_resistance_min:.1f}" if s.fire_resistance_min is not None else "   —"
            status = "PASS" if s.passes_600 else "FAIL"
            lines.append(
                f"{s.eid:>6}  {s.T_initial:>7.1f}°  {s.T_peak_centroid:>7.1f}°  "
                f"{t500:>10}  {t600:>10}  {fr:>9}  {status}"
            )
        return "\n".join(lines)

    # ── CSV export ────────────────────────────────────────────────────────────

    def to_csv(
        self,
        path: Path | str,
        T_crit: float = T_CRIT_DEFAULT,
    ) -> None:
        """
        Export per-element summary table to a CSV file.

        Columns: element_id, T_initial_C, T_peak_centroid_C, T_peak_nodal_C,
                 t_crit_500_s, t_crit_500_min, t_crit_600_s, t_crit_600_min,
                 fire_resistance_min, passes_600
        """
        path = Path(path)
        fieldnames = [
            "element_id",
            "T_initial_C",
            "T_peak_centroid_C",
            "T_peak_nodal_C",
            "t_crit_500_s",
            "t_crit_500_min",
            "t_crit_600_s",
            "t_crit_600_min",
            "fire_resistance_min",
            "passes_600",
        ]

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else ""

        summaries = self.element_summaries()
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    "element_id":          s.eid,
                    "T_initial_C":         f"{s.T_initial:.4f}",
                    "T_peak_centroid_C":   f"{s.T_peak_centroid:.4f}",
                    "T_peak_nodal_C":      _fmt(s.T_peak_nodal),
                    "t_crit_500_s":        _fmt(s.t_crit_500),
                    "t_crit_500_min":      _fmt(s.t_crit_500 / 60.0 if s.t_crit_500 is not None else None),
                    "t_crit_600_s":        _fmt(s.t_crit_600),
                    "t_crit_600_min":      _fmt(s.t_crit_600 / 60.0 if s.t_crit_600 is not None else None),
                    "fire_resistance_min": _fmt(s.fire_resistance_min),
                    "passes_600":          "True" if s.passes_600 else "False",
                })

        log.info("Summary CSV written → %s  (%d elements)", path, len(summaries))

    def to_temperature_csv(self, path: Path | str) -> None:
        """
        Export full centroid temperature time-history to a CSV file.

        Columns: time_s, time_min, T_eid_<id1>, T_eid_<id2>, …
        """
        path = Path(path)
        data = self.all_centroid_history()
        t_s   = data["times_s"]
        t_min = data["times_min"]
        eid_keys = [f"T_eid_{eid}" for eid in self._tf.element_ids]
        fieldnames = ["time_s", "time_min"] + eid_keys

        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(t_s)):
                row: dict[str, object] = {
                    "time_s":   f"{t_s[i]:.2f}",
                    "time_min": f"{t_min[i]:.4f}",
                }
                for key in eid_keys:
                    row[key] = f"{data[key][i]:.4f}"
                writer.writerow(row)

        log.info(
            "Temperature history CSV written → %s  (%d steps × %d elements)",
            path, len(t_s), self.n_elements,
        )

    # ── Pandas export ─────────────────────────────────────────────────────────

    def to_dataframe(self, T_crit: float = T_CRIT_DEFAULT) -> "pd.DataFrame":
        """
        Return the element summary as a pandas DataFrame.

        Requires ``pandas`` to be installed.

        Returns
        -------
        pd.DataFrame
            Index: element_id (int).
            Columns: T_initial_C, T_peak_centroid_C, T_peak_nodal_C,
                     t_crit_500_s, t_crit_500_min, t_crit_600_s,
                     t_crit_600_min, fire_resistance_min, passes_600.
        """
        import pandas as pd  # noqa: PLC0415

        records = []
        for s in self.element_summaries():
            records.append({
                "T_initial_C":         s.T_initial,
                "T_peak_centroid_C":   s.T_peak_centroid,
                "T_peak_nodal_C":      s.T_peak_nodal,
                "t_crit_500_s":        s.t_crit_500,
                "t_crit_500_min":      s.t_crit_500 / 60.0 if s.t_crit_500 is not None else None,
                "t_crit_600_s":        s.t_crit_600,
                "t_crit_600_min":      s.t_crit_600 / 60.0 if s.t_crit_600 is not None else None,
                "fire_resistance_min": s.fire_resistance_min,
                "passes_600":          s.passes_600,
            })

        return pd.DataFrame(records, index=pd.Index(self._tf.element_ids, name="element_id"))

    def to_history_dataframe(self) -> "pd.DataFrame":
        """
        Return the centroid temperature time-history as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Index: time_s (float).
            Column ``time_min`` plus one column per element (``T_eid_<id>``).
        """
        import pandas as pd  # noqa: PLC0415

        data = self.all_centroid_history()
        df = pd.DataFrame(index=pd.Index(data["times_s"], name="time_s"))
        df["time_min"] = data["times_min"]
        for eid in self._tf.element_ids:
            df[f"T_eid_{eid}"] = data[f"T_eid_{eid}"]
        return df
