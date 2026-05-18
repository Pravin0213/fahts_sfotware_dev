# Section Mesh Domain — FAHTS

## CRITICAL: Legacy vs. Active meshers

There are two generations of meshers. **Do not confuse them.**

| Mesher | File | Status | Used by |
|--------|------|--------|---------|
| `BoxMesher` | `box_mesher.py` | Legacy — cross-section 2D | Tests only (kept for test coverage) |
| `IProfileMesher` | `ihprofil_mesher.py` | Legacy — cross-section 2D | Tests only |
| `PipeMesher` | `pipe_mesher.py` | Legacy — cross-section 2D | Tests only |
| `BoxSurfaceMesher` | `box_surface_mesher.py` | **Active** | `analysis_runner.py` |
| `IProfileSurfaceMesher` | `iprofil_surface_mesher.py` | **Active** | `analysis_runner.py` |
| `PipeSurfaceMesher` | `pipe_surface_mesher.py` | **Active** | `analysis_runner.py` |
| `PlateSurfaceMesher` | `plate_surface_mesher.py` | **Active** | `analysis_runner.py` |
| `ShellMesher` | `shell_mesh.py` | Active (TRISHELL fallback) | `analysis_runner.py` |

**Rule:** All new solver work uses surface meshers. Legacy meshers are read-only reference.

---

## Active Surface Meshers

All surface meshers produce a `BeamSurfaceMesh`. They tile the **outer surface** of the beam
(axial × hoop plane). The heat equation is 2-D in the face plane; wall thickness is a scalar.

### BoxSurfaceMesher
```python
BoxSurfaceMesher(section: BoxSection, length: float,
                 n_top=2, n_side=3, n_length=4).build() → BeamSurfaceMesh
```
4 outer faces: bottom, top, left web, right web.
Default mesh: 2×3×4 = 40 quad elements, 50 nodes.

### IProfileSurfaceMesher
```python
IProfileSurfaceMesher(section: ISection, length: float,
                      n_top=4, n_side=2, n_bottom=2, n_length=2).build() → BeamSurfaceMesh
```
8 outer faces: top flange (top surface + 2 overhang undersides), web (left + right),
bottom flange (2 overhang undersides + bottom surface).
Corner nodes shared at (x, ±tw/2, z_top_in) and (x, ±tw/2, z_bot_in).

### PipeSurfaceMesher
```python
PipeSurfaceMesher(section: PipeSection, length: float,
                  c_circ=8, n_length=4).build() → BeamSurfaceMesh
```
Cylindrical outer surface. `local_coords_2d` = unrolled (axial × arc-length).
Total area = 2π·R_outer·length.

### PlateSurfaceMesher
```python
PlateSurfaceMesher(section: PlateSection, corners: np.ndarray,
                   mesh_12=4, mesh_14=2).build() → BeamSurfaceMesh
```
`corners`: shape (4,3) — actual structural node positions in global coords.
Bilinear grid on plate face. `local_coords_2d` = projected onto plate local (e1, e2) frame.
Used for QUADSHEL elements only. TRISHELL → ShellMesher fallback.

---

## BeamSurfaceMesh — beam_surface_mesh.py

```python
@dataclass
class BeamSurfaceMesh:
    nodes_3d: np.ndarray          # (n_nodes, 3) global coords
    quads: np.ndarray             # (n_quads, 4) node indices, CCW
    node_areas: np.ndarray        # (n_nodes,) lumped area per node [m²]
    outer_face_indices: list[int] # quad indices exposed to fire
    inner_node_indices: list[int] # node indices on inner surface (BOX/PIPE only)
    local_coords_2d: np.ndarray | None  # (n_quads, 4, 2) — used for PIPE/plate
```

`element_coords_2d(q)` — returns 2D coords for quad q:
- If `local_coords_2d` is set: returns `local_coords_2d[q]`
- Otherwise: extracts from `nodes_3d` using beam local axes

`SurfaceTransientSolver` calls `element_coords_2d()` uniformly for all section types.

---

## Legacy SectionMesh (cross-section) — section_mesh.py

Used only by legacy `TransientSolver` and legacy mesher tests.

```python
@dataclass
class SectionMesh:
    nodes: np.ndarray         # (n, 2) [y, z] in local cross-section coords
    quads: np.ndarray         # (n, 4) CCW node indices
    outer_edge_pairs: list    # edges on fire-exposed boundary
    inner_edge_pairs: list    # edges on inner (adiabatic) boundary
    inner_node_indices: list  # derived: {n for pair in inner_edge_pairs for n in pair}
```

---

## ShellMesh1D — shell_mesh.py (TRISHELL fallback)

1-D through-thickness mesh for triangular shell elements.
```python
ShellMesher(section: PlateSection, n_layers=1).build() → ShellMesh1D
ShellMesh1D.outer_node = 0      # fire-exposed face
ShellMesh1D.inner_node = n_layers  # adiabatic face
```
