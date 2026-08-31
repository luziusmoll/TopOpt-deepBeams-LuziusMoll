# CLAUDE.md

Guidance for working in this repository. This project is a **reference implementation of
density-based (SIMP) topology optimization for 2-D deep beams**, written in pure
Python/NumPy/SciPy on top of CALFEM + gmsh. A parallel research goal is to make
**Kratos Multiphysics** the FE solver (and later the optimization framework) *without
changing the JSON problem definition*. See "Kratos compatibility" below.

## Running

```bash
python src/main.py                                 # interactive: optional geometry/parameter GUIs
python src/main.py --example Examples/bridge_1      # reads <folder>/{geometry,parameters}.json, then opens the STM GUI
python src/main.py --example Examples/bridge_1 --no-gui   # fully headless: no prompts, no GUI, no plt.show(); just writes the PDFs
```

Output is always `results/optimized_structure.pdf` (grayscale density map); in
`--example` mode a copy is also written to `<folder>/optimized_structure.pdf`. Without
`--no-gui`, `strutandtieGUI` then opens so the user can trace a strut-and-tie model over
the density field, writing `<folder>/trusses.json`. Use `--no-gui` for batch runs / CI /
regenerating the Example PDFs. Dependencies: `calfem-python`, `gmsh`, `numpy`, `scipy`,
`matplotlib`, `PyQt5`, `PyMuPDF` (`.venv/` on disk, git-ignored).

## Repository layout

```
src/
  main.py             entry point / orchestration
  system_setup.py     geometry.json -> CALFEM Geometry -> GmshMesh -> Node/MembraneElement; BC application
  mesh.py             Mesh.create(): CALFEM coords/edof/dofs arrays -> object model
  node.py             Node (coords, dofs[2], forces[2], fixed[2], displacements)
  membrane_element.py MembraneElement: 4-node quad; SIMP compliance & sensitivity; cached Ke
  membrane.py         QuadPlateMembrane: bilinear plane-stress Ke, B-matrix, Jacobian, stresses
  beam_element.py     BeamElement: 2-node 3-DOF frame element  (LEGACY - strut-and-tie post-proc only)
  system.py           System: assembly, solve, objective, filter, OC loop, plotting (~960 lines)
  geometryGUI.py      tkinter canvas  -> config/geometry.json
  parameterGUI.py     tkinter form    -> config/parameters.json
  strutandtieGUI.py   overlay density PDF, draw truss -> trusses.json
  utils/config.py     load_config(): cast the 8 string params to float/int
  utils/utils.py      oc(): Optimality-Criteria update (bisection on Lagrange multiplier)
config/               empty; populated at runtime by the GUIs
Examples/<case>/      geometry.json + parameters.json (+ notes.txt, output PDFs, trusses.json)
```

Git: work happens on branch `TopOpt`; PR base is `main`. Kratos is **not** a submodule -
it lives separately at `~/Kratos`.

## Input schema (the "problem definition" - keep stable)

`parameters.json` - all values are strings, cast in `utils/config.py`:
`volfrac`, `penalty`, `x_min`, `r_min`, `Youngs_modulus`, `Poissons_ratio`,
`max_iteration`, `mesh_el_size`. Optional: `change_tol` (density-change stop tolerance,
default 0.01) - read directly by `System.__init__`, not by `load_config`.

`geometry.json` (see `Examples/README.txt` for the canonical form):
- `surfaces`: list of polygons; `surfaces[0]` = outer boundary, `surfaces[1:]` = holes
  (passed to CALFEM as `g.surface(outer_lines, [hole_loops])`).
- `load_points`: `[[x,y], [fx,fy]]`
- `load_lines`: `[[[x1,y1],[x2,y2]], [fx,fy]]`
- `support_points`: `[[x,y], [fix_x, fix_y]]`  (booleans, per DOF)
- `support_lines`: `[[[x1,y1],[x2,y2]], [fix_x, fix_y]]`

## Computational workflow

1. **Mesh** (`SystemSetup.create_mesh_from_geometry`): build CALFEM `Geometry` from
   `surfaces`; `GmshMesh` with `elType=3` (4-node quad), `dofsPerNode=2`,
   `elSizeFactor = mesh_el_size`. `Mesh.create` converts `coords/edof/dofs` (1-based ->
   0-based) into `Node` + `MembraneElement` objects.
2. **Material**: `main.py` sets `e.E`, `e.nu` on every element from parameters;
   `System.__init__` sets `e.system_penalty` and initializes `x = volfrac` uniformly.
3. **BCs** (`SystemSetup.apply_boundary_conditions`): geometric selection -
   `fix_node_by_coord` / `load_point` = nearest node; `fix_line` / `load_line` =
   point-to-segment projection with `tol=1e-4`.
4. **FE solve** (`System.solve_FE_sparse`): dense `K_global()` (`K += x_e^p * k_e_global`),
   `F_global()`, homogeneous Dirichlet by zeroing rows/cols + 1 on diagonal, then
   `scipy.sparse.linalg.spsolve`; displacements written back onto nodes/elements.
5. **Objective** `System.compliance()` = sum of `x_e^p * u_e^T k0 u_e` (Sigmund 2001 eq.1).
6. **Sensitivity** `System.sensitivity_compliance()` = `-p * x_e^(p-1) * (u_e^T k0 u_e)` (eq.4).
7. **Filter** `System.convolution_operator()` builds a dense NxN weight matrix
   `H_f[i,j] = max(0, r_min - dist(centroid_i, centroid_j))`; sensitivity is then smoothed
   in `top_opt` per Sigmund eq. 5 (fixed in `eaa8e42`; history in B1/B2 below).
8. **Update** `oc()` - OC with move limit 0.2, damping 1/2; area-weighted volume check
   `sum(A_e x_e) > volfrac * sum(A_e)` (B4 fix), `dv` = per-element areas.
9. **Loop** `System.top_opt`: stop on max density change `< change_tol` (default 0.01) or
   `max_iteration` (B3 fix - Sigmund's rule).
10. **Output** `System.plot2()` -> density PDF; then `TrussInputGUI` -> `trusses.json`.

## Mathematical formulation

Minimum-compliance SIMP, 2-D plane stress:
`min_x  c(x) = U^T K(x) U = sum_e x_e^p * u_e^T k0 u_e`
s.t. `K(x) U = F`, `(1/N) sum_e x_e <= volfrac`, `x_min <= x_e <= 1`.
Element: 4-node bilinear quad, 2x2 Gauss, plane stress
`D = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]`, unit thickness.
Filter: Sigmund mesh-independency sensitivity filter (linear hat weights), eq. 5/6
(implemented per eq. 5 since `eaa8e42`; see "Deviations from Sigmund 2001" for history).
Update: Optimality Criteria. References in code: Sigmund (2001) "99 line"
(`reference/sigmund2001.pdf`); DTU 200-line Python code; Xia, Langelaar & Hendriks (2020)
for the downstream STM evaluation.

## Deviations from Sigmund 2001 (parity-relevant)

The code cites Sigmund, O. (2001) "A 99 line topology optimization code written in Matlab",
*Struct Multidisc Optim* 21:120-127 (`reference/sigmund2001.pdf`). What matches the paper
exactly: material law `K = sum xe^p k0` with bare power law and no `xmin+(1-xmin)xe^p` floor
(`system.py:46`); objective eq. 1 (`membrane_element.py:78-81`); sensitivity eq. 4
(`membrane_element.py:91-94`); weight kernel eq. 6 `Hf = max(0, r_min - dist(e,f))` on
centroid distance (`system.py:232-234`); OC update eq. 2/3 with `move=0.2`, `eta=1/2`,
lambda by bisection (`utils/utils.py:16-25`); uniform start `x = volfrac` (`system.py:21`);
Q4 bilinear plane-stress element (`membrane.py`).

Status: **B1-B4 all fixed** - B1/B2 in `eaa8e42` ("Fix sensitivity filter to match Sigmund
2001 eq. 5"), B3/B4 in `53e7b33` ("Fix B3 (convergence criterion) and B4 (area-weighted
volume)"). B5 is cosmetic and left as-is. The committed
`Examples/*/optimized_structure.pdf` predate these fixes and are no longer a valid
regression baseline - regenerate before relying on them.

- **B1 - sensitivity-filter normalization (FIXED in `eaa8e42`)**. Eq. 5 is
  `dc_hat_e = 1/(xe * sum_f Hf) * sum_f (Hf * xf * dc_f)`; appendix line 62 divides by
  `x(j,i)*sum`. The old code (`1/x[i] * sum(H_f[:,i]) * sum(H_f[:,i]*x*dc)`) *multiplied* by
  `S_i = sum_f Hf[f,i]` instead of dividing - off by `S_i^2`, which (since `S_i` is smaller
  near the domain boundary, not a global constant the OC lambda-bisection could absorb)
  down-weighted boundary elements by `~S_i^2` and thinned the structure at free edges.
  `system.py:273` now reads `1 / (x[i] * np.sum(H_f[:, i])) * np.sum(H_f[:, i] * x * dc)`,
  i.e. eq. 5 as written. The `if x[i]*np.sum(H_f[:,i]) > 0` guard is still always true
  (so the `else: dc_filtered_i = dc[i]` branch is dead), but it now also guards the divide.
- **B2 - filter density field (FIXED in `eaa8e42`)**. Previously `x = self.x.copy()` was
  taken once *before* the loop and never refreshed, so the filter weighted by the initial
  uniform `volfrac` field for the whole run while FE/compliance/sensitivity used the live
  `self.x`. `system.py:258` now re-snapshots `x = self.x.copy()` each iteration right after
  the solve, matching Sigmund's `check` (which uses the current `x` in both the `xf*dc_f`
  weighting and the `1/xe` normalization). The pre-loop copy at `system.py:249` is now
  redundant but harmless (`xold` is initialised separately at line 250).
- **B3 - convergence criterion (FIXED in `53e7b33`)**. Sigmund stops on design change
  `max(abs(x-xold)) < 0.01` (sec. 3.1). The old code compared `x` against a copy of itself
  (the OC update lands in `self.x`, not `x`), so `change` was `0` from iteration 2 on and
  unused; the loop ran on `obj_change < 1e-6` OR `max_iteration` (default 50), a tighter,
  design-decoupled test that usually terminated on the iteration cap. Now `top_opt` runs
  `while change > self.change_tol and loop < max_iteration` with
  `change = norm(self.x - x, inf)` (`x` = the pre-update snapshot taken at `system.py:261`).
  `self.change_tol` defaults to `0.01` and is read from `parameters.json` only if a
  `change_tol` key is present (`system.py` `__init__`); no schema/GUI change. The
  `obj_change` machinery and `xold` are removed; `obj_hist` is still recorded for plotting.
- **B4 - element-area weighting (FIXED in `53e7b33`)**. The old code passed
  `dv = np.ones(n_el)` into `top_opt` and `oc` tested `np.mean(xnew) > volfrac`, i.e.
  `dV/dxe = 1` and an unweighted volume fraction - wrong on an unstructured mesh with
  unequal element sizes. Now `main.py` passes `dv = system.sensitivity_densitiy()`
  (per-element shoelace areas) and `oc` (`utils/utils.py:21`) tests
  `np.sum(dv * xnew) > volfrac * np.sum(dv)`. The OC update already divided by `dv`, so
  `Be = -dc_e / (lambda * A_e)` now falls out correctly. Verified on an 8x4 test mesh
  (element areas 0.11-0.29): area-weighted volume fraction converges to 0.4000 vs an
  unweighted mean of 0.4076 under the old code. Near-identical on regular grids.
- **B5 - minor**: OC bisection uses a relative stop `(l2-l1)/(l1+l2) > 1e-8` plus an
  `if l1+l2==0: return` guard (`utils/utils.py:16,27-28`) vs Sigmund's absolute
  `l2-l1 > 1e-4`; `l2` init `1e9` vs `1e5` - functionally equivalent. Dirichlet BCs are
  applied by zeroing rows/cols + 1 on the diagonal with `F=0` (`system.py:75-83`) instead
  of eliminating fixed DOFs - equivalent for homogeneous BCs only.
- **Not carried over from the paper**: multiple load cases (sec. 4.2), passive/active
  element masks (sec. 4.3 - true mesh holes are used instead), penalty continuation
  (Sigmund also uses fixed `p`).

### Parity mode checklist (when validating a Kratos port against the current code)

The code *as of `53e7b33`* is now a faithful Sigmund implementation except for: the bare
`x^p` interpolation with no stiffness floor, the dense (unwindowed) filter, and the
mesh-dependent `load_line` (equal nodal force on every edge node). A Kratos port that
matches those three and uses eq. 5 filtering, the live density field, design-change
convergence, and area-weighted volume will track the current Python results.
To instead reproduce the *older committed Example PDFs* (pre-`eaa8e42`), also re-introduce
B1 (multiply by `sum(Hf)`), B2 (freeze the filter weighting at `volfrac`), B3 (`obj_change`
+ iteration-cap stop), and B4 (`dv = 1`, unweighted `mean(x)` volume test).

## Other quirks / gotchas

- **`load_line` is mesh-dependent**: it *overwrites* `node.forces` with the full force
  vector on *every* node of the edge (`system.py:382-399`), so the total applied load grows
  with refinement. For compliance minimization on a uniform edge mesh this only rescales
  `c` (topology unaffected), but it is not a consistent nodal load.
- **Only homogeneous Dirichlet BCs** are supported (matches the boolean `[fix_x,fix_y]`
  schema).
- **`nr_dofs = nodes[-1].dofs[-1] + 1`** assumes contiguous node numbering.
- **Dense global stiffness**: `K_global()` allocates `nr_dofs x nr_dofs` (`system.py:41`)
  before CSR conversion; `convolution_operator()` builds a dense NxN (`system.py:232`) with
  no search-window restriction. Both are O(N^2) - fine for teaching-scale meshes only, and
  the real scalability ceiling.
- **Legacy / not on the optimization path** (no need to maintain for topology runs):
  the `regular_mesh` analytic-KE branch in `MembraneElement.k_e_global`
  (`membrane_element.py:41-59`, also references a non-existent `self.k_e` - should be
  `self.k_e_matrix` - but `Mesh.create` always passes `regular_mesh=False`), `BeamElement`,
  and the frame/STM methods in `system.py` (`strain_energy_beam_truss`,
  `recover_internal_forces`, `sts`, `plot_deformed_stm_sf*`).

## Kratos compatibility (research direction)

Goal: keep `geometry.json` + `parameters.json` as the single source of truth while Kratos
replaces the FE solver, and later the whole optimizer, behind a thin adapter. The JSON is
already solver-agnostic; nothing in it is CALFEM-specific.

### Kratos build on this machine (`~/Kratos/bin/Release`, already on PYTHONPATH)

- **Compiled**: `StructuralMechanicsApplication`, `ConstitutiveLawsApplication`,
  `LinearSolversApplication`, `FluidDynamicsApplication`, `IgaApplication`, `RomApplication`.
- **Source present but NOT compiled** (would need a rebuild - do not attempt without being
  asked): `OptimizationApplication`, `TopologyOptimizationApplication`,
  `ShapeOptimizationApplication`, `MeshingApplication`, `MedApplication`.
- Verified names in `StructuralMechanicsApplication`: `SmallDisplacementElement2D4N`,
  `SmallDisplacementElement2D3N`, `LinearElasticPlaneStress2DLaw`,
  `LinearElasticPlaneStrain2DLaw`, `PointLoadCondition2D1N`, `LineLoadCondition2D2N`.

### Component mapping

| Stage | StructuralMechanicsApplication (available now) | TopologyOptimizationApplication / OptimizationApplication (if enabled) |
|---|---|---|
| mesh from geometry.json (+ holes) | keep CALFEM/gmsh; build `ModelPart` in memory (`CreateNewNode`/`CreateNewElement`) or emit `.mdpa` | same mesh source; consumes a standard `ModelPart` / `.mdpa` |
| `MembraneElement` | `SmallDisplacementElement2D4N` + `LinearElasticPlaneStress2DLaw` + `THICKNESS=1.0` | `SmallDisplacementSIMPElement` (carries `X_PHYS`, computes `DCDX`/`DVDX`) |
| `e.E`, `e.nu` | `Properties`: `YOUNG_MODULUS`, `POISSON_RATIO` | idem; penalty/E_min inside the SIMP element |
| `K += x^p k_e` | per-element `Properties`, set `YOUNG_MODULUS = E0*(x_min+(1-x_min)*x_e^p)` each iter | native SIMP/RAMP interpolation |
| solve / Dirichlet / `spsolve` | `ResidualBasedLinearStrategy` + block builder + `LinearSolversApplication` solver, or the `StructuralMechanicsAnalysis` stage | `topology_optimization_simp_static_solver.py` |
| supports (per-DOF flags) | replicate `fix_line`/nearest-node selection -> sub-model-part -> fix `DISPLACEMENT_X/_Y` | same via `ProjectParameters` processes |
| point loads | `PointLoadCondition2D1N` + `POINT_LOAD` | idem |
| line loads | parity: equal nodal `POINT_LOAD` on each edge node; physical: `LineLoadCondition2D2N` + `LINE_LOAD` | idem |
| `compliance()` | sum of elemental `STRAIN_ENERGY`, or `0.5 U^T F` | `structure_response_function_utilities.h` |
| `sensitivity_compliance()` | from elemental `U_e`: `dc_e = -2 p U_e / x_e` (no extra assembly) | native `DCDX` / adjoint sensitivity strategy |
| `convolution_operator()` + filter | keep existing NumPy `H_f` filter (centroid math, solver-agnostic) | `topology_filtering_utilities.h`, or Helmholtz/vertex-morphing filter in OptimizationApplication |
| `oc()` | keep existing NumPy OC | `topology_updating_utilities.h`, or OptimizationApplication algorithms (OC / MMA / gradient projection) |
| `top_opt()` loop | keep as driver; only FE solve + energy come from Kratos | replaced wholesale by `topology_optimizer_factory.SIMPMethod` |
| `plot2()` + `trusses.json` STM GUI | unchanged - feed the per-element density array back into the matplotlib code | idem; optionally also Kratos VTK/GiD `X_PHYS` output |

### Phased plan

1. **Kratos as FE solver only** (`StructuralMechanicsApplication`, no rebuild): adapter
   builds a `ModelPart` from the existing gmsh mesh; each OC iteration writes
   `YOUNG_MODULUS` per element, solves, reads elemental strain energy for objective +
   `dc_e = -2 p U_e / x_e`. Filter, `oc()`, plotting, STM GUI unchanged. Validate against
   current `spsolve` displacements and the committed Example PDFs.
2. **Kratos as topology framework** (`TopologyOptimizationApplication`, needs enabling):
   adapter emits `.mdpa` + `ProjectParameters.json` + optimizer config (objectives =
   strain energy, constraints = volume fraction, `r_min`, `penalty`, `max_iteration`);
   `SIMPMethod` runs the loop; pull final `X_PHYS` back into `plot2()`.
3. **Modern framework** (`OptimizationApplication`): Helmholtz filtering, MMA/gradient
   projection, multi-load-case / stress constraints. Same input files.

### Adapter caveats (behavioural differences vs. the current code)

- Decide per phase whether to **replicate** the mesh-dependent `load_line` semantics and
  the `dv=1` / `mean(x)>volfrac` volume check for numerical parity, or switch to
  physically consistent line loads and true element volumes.
- Any switch to Kratos/standard sensitivity filtering will not reproduce the current
  results bit-for-bit (see "Known quirks" - the filter formula and the missing SIMP
  stiffness floor). Keep a "parity mode" if exact reproduction is required.
- Holes need no special handling - gmsh meshes them; Kratos only sees the final mesh.
- Do not modify, build, or install into the `~/Kratos` tree unless explicitly asked.

## Constraints when working here

- The working tree has pre-existing uncommitted user changes (`src/system.py`,
  `Examples/cantilever1/parameters.json`, `results/optimized_structure.pdf`, plus untracked
  PDFs / `trusses.json` / `run_all_examples.sh`). Do not revert, stage, or commit them.
- Commit / push only when asked; branch off `TopOpt` if you do.
