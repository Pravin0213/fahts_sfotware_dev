"""Command-line runner for vessel cases (no GUI).

    fahts-run case.vcase.json                     # run, print the summary
    fahts-run path/to/vessfire_deck --out r.xlsx   # import a VessFire deck, run, write Excel
    fahts-run case.vcase.json --out series.csv --t-end 1800
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from fahts.coupling import VesselCase, run_case
from fahts.coupling.report import summary_lines
from fahts.coupling.vessfire_import import case_from_vessfire_deck


def load(path: Path) -> tuple[VesselCase, list[str]]:
    if path.is_dir():
        return case_from_vessfire_deck(path)
    return VesselCase.load(path), []


def write(result, out: Path) -> None:
    if out.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(out) as xw:
            pd.DataFrame(summary_lines(result), columns=["item", "value"]).to_excel(
                xw, sheet_name="summary", index=False)
            result.series.to_excel(xw, sheet_name="time series", index=False)
            result.failures.to_excel(xw, sheet_name="failure times", index=False)
            result.stress.to_excel(xw, sheet_name="stress", index=False)
    else:
        result.series.to_csv(out, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fahts-run", description="Run a vessel-in-fire case.")
    ap.add_argument("case", type=Path, help="*.vcase.json file or VessFire input-deck folder")
    ap.add_argument("--out", type=Path, help="write results (.xlsx: all sheets, else CSV series)")
    ap.add_argument("--t-end", type=float, help="override the end time [s]")
    ap.add_argument("--save-case", type=Path, help="also save the (imported) case as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        case, warnings = load(args.case)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read {args.case}: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if args.t_end:
        case.run.t_end_s = args.t_end
    if args.save_case:
        case.save(args.save_case)
    errors = case.validate()
    if errors:
        print("error: invalid case:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    def progress(t, t_end):
        if not args.quiet:
            print(f"\r  t = {t:7.0f} / {t_end:.0f} s", end="", file=sys.stderr, flush=True)

    result = run_case(case, progress=progress)
    if not args.quiet:
        print(file=sys.stderr)
    width = max(len(k) for k, _ in summary_lines(result))
    for k, v in summary_lines(result):
        print(f"{k:<{width}}  {v}")
    if args.out:
        write(result, args.out)
        print(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
