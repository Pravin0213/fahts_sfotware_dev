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

```
FAHTS_solver/
├── CLAUDE.md              ← THIS FILE
├── ROADMAP.md             ← Full product plan and phase breakdown
├── docs/
│   ├── phase_history.md   ← Completed phase details (archive — not auto-loaded)
│   └── 3D_FEM_heat_transfer_theory.txt  ← Solver theory reference
├── main.py                ← Entry point
├── model_file.fem         ← Primary test model (659 nodes, ~783 beams, BOX only)
├── model_t1.fem           ← Extended model (IHPROFIL + PIPE + BOX + shells)
├── model_t3.fem           ← Asymmetric BOX sections
├── usfos_verification_results/  ← Benchmark reference (fahts_beltemp.fem)
├── legacy/                ← READ-ONLY reference scripts, do NOT import
│
└── fahts/
    ├── core/model/        ← FEMModel, Node, BeamElement, BoxSection, ISection,
    │                         PipeSection, PlateSection, SteelMaterial, Group
    ├── core/model/        ← FEMModel, BeamElement, BoxSection, ISection,
    │   │                     PipeSection, PlateSection, SteelMaterial, Node, Group
    │   └── CLAUDE.md      ← Section field names, UNITVEC convention, dataclass contracts
    ├── core/io/           ← usfos_reader.py, results_writer.py, beltemp_parser.py
    │   └── CLAUDE.md      ← USFOS format, BELTEMP format, benchmark config
    ├── core/heat/
    │   ├── sources/       ← fire_zone.py, rad_ball.py
    │   ├── bc/            ← view_factor.py, net_flux.py
    │   ├── section_mesh/  ← BoxSurfaceMesher, IProfileSurfaceMesher,
    │   │   │                 PipeSurfaceMesher, PlateSurfaceMesher,
    │   │   │                 BeamSurfaceMesh (+ legacy cross-section meshers)
    │   │   └── CLAUDE.md  ← Legacy vs active meshers, BeamSurfaceMesh fields
    │   └── solver/        ← surface_solver.py (primary), analysis_runner.py,
    │       │                 time_integrator.py (legacy), shell_1d_solver.py
    │       └── CLAUDE.md  ← Solver architecture, CN equations, dispatch logic
    ├── core/results/      ← temperature_field.py, analysis_config.py, post_processor.py
    │   └── CLAUDE.md      ← TemperatureField/AnalysisConfig/PostProcessor APIs, results flow
    ├── renderer/          ← beam_geometry.py, scene_manager.py, colormap.py
    └── gui/
        ├── CLAUDE.md      ← Qt patterns, panels, dialogs, animation toolbar
        ├── main_window.py
        ├── analysis_worker.py
        ├── panels/
        └── dialogs/

tests/
└── CLAUDE.md              ← Test conventions, count, run command
```

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
| **Phase 3E.8 — USFOS Benchmark** | **⬜ Next task** |
| Phase 5 — Insulation + Advanced | ⬜ Not started |

**Tests: 1008 passing, 1 skipped.**
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

`build_model_mesh` produces **USFOS-style thin panels** (no wall thickness rendered):
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
python main.py                 # empty app
python main.py model_file.fem  # BOX-only model
python main.py model_t1.fem    # mixed sections
python -m pytest tests/ -q     # run all tests
```
