# FAHTS — Agent Handover Brief

**FAHTS** (Fire Analysis and Heat Transfer Software) is a desktop app for 3D heat transfer
analysis of structural steel under fire. It reads USFOS `.fem` models, renders them in 3D
(PyQt6 + PyVista), places fire zones in the scene, and solves the heat equation in each
steel member to produce temperature-time curves.

Theory source: `FAHTS_theory/Fahts_Theory_Manual.pdf` (Ch. 3).
Full theory notes: `docs/3D_FEM_heat_transfer_theory.txt`.
Completed phase details: `docs/phase_history.md`.

---

## Architectural Decisions (DO NOT CHANGE without strong reason)

| Decision | Choice | Why |
|----------|--------|-----|
| UI framework | PyQt6 + pyvistaqt | Desktop app; PyVista for 3D |
| All section solvers | 3-D surface mesh, axial × hoop | Matches SINTEF FAHTS original |
| Time integration | Crank-Nicolson θ=1/2 | Unconditionally stable, 2nd-order; matches FAHTS |
| File format | USFOS .fem | Project requirement |
| Steel properties | EN 1993-1-2 Annex C | Eurocode standard |
| Coordinate system | Global vessel frame, metres | model_file.fem uses X≈350m |
| Multi-zone BC | max(T_fire) per element | Hottest covering zone wins |
| BOX mesh defaults | n_top=2, n_side=3, n_length=4 | Matches FAHTS defaults |

---

## Project Layout

The repo is being restructured (branch `restructure-monorepo`, started 2026-09-29) into a
product that couples the structural heat solver with a process-equipment model (pressure,
relief, rupture). ✅ = exists now, ⬜ = planned (ported from `Test/vfpy`).

```
vessfire_heatsolver/
├── CLAUDE.md  ROADMAP.md  pyproject.toml   ← pip install -e ".[gui,process,dev]"
├── main.py                ← thin launcher shim → src/fahts/__main__.py
├── src/fahts/             ✅ the product — only this ships
│   ├── __main__.py        ✅ GUI entry point (python -m fahts / `fahts`)
│   ├── core/model/        ✅ FEMModel, sections, SteelMaterial (see its CLAUDE.md)
│   ├── core/io/           ✅ usfos_reader, results_writer, beltemp_parser
│   ├── core/heat/         ✅ structural heat solver: sources/, bc/, radiation/,
│   │                         section_mesh/, solid_mesh/, solver/ (→ becomes wall/fem_3d/)
│   ├── core/results/      ✅ TemperatureField, AnalysisConfig, PostProcessor
│   ├── renderer/  gui/    ✅ 3-D view + Qt app
│   ├── common/            ⬜ units, constants, errors, result containers
│   ├── materials/         ⬜ steel k/cp/ρ(T) + strength/E/α(T) (merge SteelMaterial + vfpy Material)
│   ├── thermo/            ⬜ PR EOS + flash, pseudo-components, CoolProp adapter
│   ├── fire/              ⬜ fire loads shared by all solvers (zones, rad-ball, black-body BC)
│   ├── wall/column_1d/    ⬜ radial 1-D wall column (vfpy WallColumn)
│   ├── process/           ⬜ vessel geometry, inner-wall heat transfer, gas/liquid zones, model loop
│   ├── relief/            ⬜ BDV / PSV / orifice / line flow
│   ├── rupture/           ⬜ stresses + failure criteria + time to rupture
│   ├── coupling/          ⬜ the ONLY place that combines wall ↔ process ↔ fire
│   └── cli.py             ⬜ headless case runner
├── tests/                 ✅ (target: unit/<pkg>/, integration/, regression/)
├── validation/            ✅ runnable benchmarks + reports
│   ├── validate_3d.py     ✅ analytical checks  ├── openfoam/ ✅  ├── usfos/reference/ ✅
│   └── vessfire/          ⬜ VessFire comparison readers/tools (never imported by src/)
├── examples/models/       ✅ *.fem models (model_file, model_t1, model_t3, tank_horizontal, …)
├── studies/               ⬜ dated research scripts, never imported by src/
├── tools/vessfire_runner/ ⬜ case-matrix builders + batch runner
├── docs/                  ✅ theory, plans (target: theory/, decisions/)
├── data/                  gitignored — large reference results, study outputs
├── Test/                  gitignored — raw import being migrated; read-only source, do not edit
└── legacy/                READ-ONLY reference scripts, do NOT import
```

**Layering rule (target):** `common → materials/thermo → fire/wall/process/relief/rupture
→ coupling → cli/gui`. Physics packages never import each other; only `coupling/` combines
them. Wall ↔ process talk through one small interface (inner-wall T per region ↔ h, T_fluid
per region) so `wall/column_1d` and the 3-D FEM wall are interchangeable.

**Licence / IP:** everything derived from VessFire (reference results, runner, comparisons)
stays in `validation/`, `tools/`, `data/` — never in `src/`. Client project data and
third-party PDFs never go in git.

---

## Data Contracts (do not rename fields)

```python
# FEMModel — fem_model.py
nodes: dict[int, Node]
elements: dict[int, BeamElement]
shell_elements: dict[int, ShellElement]
sections: dict[int, Section]       # keyed by geom_id
materials: dict[int, SteelMaterial]
groups: dict[str, Group]
unitvecs: dict[int, np.ndarray]    # shape (3,)
source_file: Path

# BeamElement — element.py
eid, n1, n2, mat_id, geom_id, lcoor_id
length: float; direction: np.ndarray; local_z: np.ndarray
ecc1: np.ndarray | None; ecc2: np.ndarray | None

# TemperatureField — temperature_field.py
times: np.ndarray              # (n_steps,)
element_ids: list[int]
T_centroid: np.ndarray         # (n_steps, n_elems)
T_section: dict[int, np.ndarray]  # {eid: (n_steps, n_nodes)}

# AnalysisConfig — analysis_config.py
t_end, dt, output_dt: float
n_layers: int = 1; elem_size: float | None = None
element_ids: list[int]
# Surface mesh params: n_top/n_side/n_length (BOX), n_top_i/n_side_i/n_bottom_i/n_length_i (I),
#                      c_circ/n_length_p (PIPE), mesh_12/mesh_14 (Shell)
```

---

## Current Status (2026-05-31)

| Phase | Status |
|-------|--------|
| Phase 1 — Foundation | ✅ Complete |
| Phase 2 — Heat Source Engine | ✅ Complete |
| Phase 3 — Heat Transfer Solver | ✅ Complete (incl. 3E + 3F) |
| Phase 4 — Results Visualisation | ✅ Complete |
| **USFOS-style thin rendering** | **✅ Done** — BOX/PIPE/I-beam rendered as flat mid-surface panels; no wall thickness |
| **Mesh Inspector** | **✅ Done** — "Inspect Mesh" toggle (shortcut I); click quad → see K-matrix neighbours |
| **3-D Hex8 solid solver (2026-09-27)** | **✅ Default path** — `solver_dim="3d"`; plan + status in `docs/3d_solver_plan.md`; validation in `validation/report_3d.md` (`python -m validation.validate_3d`) |
| Phase 3E.8 — USFOS Benchmark | ⬜ Dropped (project pivot to process equipment) |
| Phase 5 — Insulation + Advanced | ⬜ Not started |

**Tests (2026-09-27, non-GUI): 944 passing, 6 pre-existing failures** (list in `docs/3d_solver_plan.md`). GUI/VTK tests abort with `QT_QPA_PLATFORM=offscreen`; run them with `QT_QPA_PLATFORM=xcb` on a real display (501 pass, 13 pre-existing failures: colormap default + animation toolbar).
Run: `python -m pytest tests/ -q`

---

## Style Guide

- Python 3.11 type hints everywhere
- `@dataclass` for data containers; `ABC`/`@abstractmethod` for base classes
- No global state; no `print()` in library code (use `logging`)
- Physics functions pure (no side effects)
- One class per file; max line length 100 chars

---

## Rendering

**Default (2026-09-27): real wall thickness** — View → *Show Wall Thickness* (shortcut `T`,
on by default) draws members from the 3-D solid meshers (outer + inner surfaces + end rings;
plates ± t/2): `build_model_mesh(..., show_thickness=True)`, `build_thick_member_mesh`,
`build_thick_shell_mesh`, `SceneManager.set_show_thickness()`. Toggle off for the legacy view.

With thickness off, `build_model_mesh` produces **USFOS-style thin panels** (no wall thickness):
- BOX → 4 lateral quads (no end caps); n_cells = 4 per beam
- PIPE → outer ring quads only; n_cells = c_circ per pipe
- I-beam → 3 flat panels (top flange / web / bottom flange); n_cells = 3 per I-beam

FEM mesh nodes sit on the same surfaces → inspector overlay never buried inside geometry.

The **Mesh Inspector** (View → Inspect Mesh, shortcut `I`) overlays the FEM surface mesh in
cyan wireframe. Click any quad to see its 4 local node indices and the K-matrix neighbours
(other quads in the same beam that share each node). Panel: `MeshInspectorPanel` in left sidebar.
`build_mesh_inspector_data(model, centroid, config=None)` builds the data; cached in `MainWindow._mesh_inspector_data`.

---

## Critical Gotchas

- **FireZone fields:** `center` (not `centre`), `dims` (not `dimensions`)
- **FireCurveType:** `ISO_834`, `HYDROCARBON`, `USER_DEFINED`
- **USERFLUX / RadiationBall:** format is `USERFLUX 0 set cx cy cz r1 flux1 r2 flux2` — no h_conv/T_env. Set `epsilon_m=0` in solver.
- **BELTEMP values are INCREMENTAL** — accumulate from T_initial=20°C
- **UNITVEC** defines local z-axis of beam. `local_y = cross(local_x, UNITVEC)`, `local_z = cross(local_y, local_x)`
- `model_file.fem` and `model_t1.fem` must always parse cleanly — smoke tests exist
- **I-beam mesh topology:** Flanges are meshed at their **inner** faces (z = z_top_in / z_bot_in), not outer. Web is at y = **−tw/2** (not y=0 — `np.sign(0)=0` would zero the outward normal and kill RadiationBall flux). The `_flange_ys` helper forces y=−tw/2 into the flange grid so gid() creates shared T-junction nodes; this gives web↔flange heat conduction in K. See `iprofil_surface_mesher.py`. Each plate has only ONE stored normal but is physically exposed on both faces (§3.4.1) — directional sources must check both `normal` and `-normal` (`double_sided` flag in `analysis_runner.py`, fixed 2026-08-25) or they silently zero out plates facing the "wrong" way. See `heat/solver/CLAUDE.md`.
- **Mesh Inspector node labels:** `pv.PolyData.extract_cells()` returns points sorted by ascending global point index, NOT in the quad's connectivity order. Node label positions must be looked up as `mesh.points[node_idx + beam_node_offset[eid]]` (via `MeshInspectorData.beam_node_offset`). Never use `highlighted.points` directly for label coordinates — labels will land at wrong corners.

---

## Agent Instructions

1. Read this file first, then the relevant subdirectory CLAUDE.md for your domain.
2. Run `python -m pytest tests/ -q` before declaring any task done.
3. Never modify `legacy/` files.
4. Mark completed tasks `✅ Done` in ROADMAP.md.
5. For solver/FEM theory: read `docs/3D_FEM_heat_transfer_theory.txt`.
6. For completed phase implementation details: read `docs/phase_history.md`.

---

## How to Run

```bash
pip install -e ".[gui,process,dev]"               # once (editable install)
python main.py                                    # empty app (or: python -m fahts)
python main.py examples/models/model_file.fem     # BOX-only model
python main.py examples/models/model_t1.fem       # mixed sections
python -m pytest tests/ -q                        # run all tests
```
