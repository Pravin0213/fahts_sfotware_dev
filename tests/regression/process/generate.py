"""(Re)generate the process-model golden files from the frozen vfpy snapshot.

    python -m tests.regression.process.generate            # all runs
    python -m tests.regression.process.generate M06-0003   # only runs of these cases

Only regenerate when the *reference* is meant to change (e.g. a deliberate physics fix
that is also applied to legacy/vfpy — which should be rare and explained in the commit).
Never regenerate to make a failing port pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import platform
import sys
import time
from datetime import date

from tests.regression.process import harness as h

log = logging.getLogger(__name__)


def _one(gr: h.GoldenRun) -> dict:
    t0 = time.perf_counter()
    ts, failures = h.run(gr, "legacy")
    ts_path, rup_path = h.golden_paths(gr)
    ts.to_csv(ts_path, index=False, float_format="%.17g")
    failures.to_csv(rup_path, index=False, float_format="%.17g")
    return dict(run=gr.name, rows=len(ts), seconds=round(time.perf_counter() - t0, 1))


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _versions() -> dict:
    import CoolProp
    import numpy
    import pandas
    import scipy
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, pandas=pandas.__version__,
                coolprop=CoolProp.__version__)


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not h.have_material_db():
        sys.exit(f"material database missing: {h.MATERIAL_DB}")
    runs = [r for r in h.RUNS if not argv or r.case_id in argv]
    h.GOLDEN_DIR.mkdir(exist_ok=True)
    with mp.Pool(min(len(runs), mp.cpu_count())) as pool:
        done = pool.map(_one, runs)
    for d in done:
        log.info("%-32s %4d rows  %6.1f s", d["run"], d["rows"], d["seconds"])

    manifest_path = h.GOLDEN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        generated=str(date.today()),
        reference="legacy/vfpy (frozen snapshot of Test/vfpy, 2026-09-24)",
        legacy_md5={p.name: _md5(p) for p in sorted(h.LEGACY_VFPY.glob("*.py"))},
        material_db_md5=_md5(h.MATERIAL_DB),
        versions=_versions(),
        out_every_s=h.OUT_EVERY,
        profiles=h.PROFILES,
        cases=h.CASES,
    )
    manifest.setdefault("runs", {}).update({d["run"]: d for d in done})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
