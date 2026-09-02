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
  pde_filter.py       HelmholtzFilter: scalar Q4 Helmholtz/PDE density filter (filter="helmholtz")
  beam_element.py     BeamElement: 2-node 3-DOF frame element  (LEGACY - strut-and-tie post-proc only)
  system.py           System: assembly, solve, objective, filter(s), OC loop, plotting
  geometryGUI.py      tkinter canvas  -> config/geometry.json
  parameterGUI.py     tkinter form    -> config/parameters.json
  strutandtieGUI.py   overlay density PDF, draw truss -> trusses.json
  utils/config.py     load_config(): cast the 8 string params to float/int
  utils/utils.py      oc(): Optimality-Criteria update (bisection on Lagrange multiplier)
config/               empty; populated at runtime by the GUIs
Examples/<case>/      geometry.json + parameters.json (+ notes.txt, output PDFs, trusses.json)
```

Git: `TopOpt` is the trunk. Feature work is done on short-lived branches off `TopOpt`
and PR'd back into `TopOpt` (e.g. `sparse-assembly` -> PR #3, `kratos-fe-adapter` -> PR
#4); `TopOpt` -> `main` is for releases. Kratos is **not** a submodule - it lives
separately at `~/Kratos`.

## Input schema (the "problem definition" - keep stable)

`parameters.json` - all values are strings, cast in `utils/config.py`:
`volfrac`, `penalty`, `x_min`, `r_min`, `Youngs_modulus`, `Poissons_ratio`,
`max_iteration`, `mesh_el_size`. Optional keys, all read directly by `System.__init__`
(not by `load_config`), all with hardcoded defaults so no schema/GUI change:
- `change_tol` - density-change stop tolerance (default 0.01).
- `filter` - `"sensitivity"` (default; Sigmund eq. 5), `"density"` (volume-weighted
  linear density filter + tanh projection) or `"helmholtz"` (PDE filter + tanh
  projection). See "Filtering options" below.
- `eta` (0.5), `beta` (1.0), `beta_max` (16.0), `beta_iter` (25), `oc_move` (0.1) -
  only used by the `"density"` / `"helmholtz"` projected path: projection threshold,
  initial/max projection sharpness, iterations between beta doublings, and the OC move
  limit for that path. **`beta = 0` disables the Heaviside projection** (physical field
  = the plain linear filter of the design variable): grey boundaries, but converges on
  the normal design-change criterion with no OC+Heaviside limit cycle - the better
  choice for load cases where the projected path oscillates.
- `solver` - `"native"` (default; in-repo sparse assembly + spsolve) or `"kratos_fe"` (Kratos
  StructuralMechanicsApplication as the FE solver only - persistent ModelPart,
  per-element `YOUNG_MODULUS = E0*x^p`, `sparse_lu`; objective/sensitivity/filter/OC stay
  in-repo). Requires Kratos on `PYTHONPATH`. See "Kratos compatibility".

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
4. **FE solve** (`System._solve_fe` -> `solve_FE_csr`): `K_global_csr()` assembles COO
   triplets straight into CSR (no dense array), `K = sum_e x_e^p k_e`; homogeneous
   Dirichlet by free/fixed DOF partition; `spsolve` on the reduced block; displacements
   written back onto nodes/elements. (`solver="kratos_fe"` swaps in the Kratos FE solve;
   see step's note and "Kratos compatibility".)
5. **Objective** `System.compliance()` = sum of `x_e^p * u_e^T k0 u_e` (Sigmund 2001 eq.1).
6. **Sensitivity** `System.sensitivity_compliance()` = `-p * x_e^(p-1) * (u_e^T k0 u_e)` (eq.4).
7. **Filter** (per `filter` param, see "Filtering options"): `"sensitivity"` (default)
   builds the dense NxN `convolution_operator()` `H_f[i,j] = max(0, r_min -
   dist(centroid_i, centroid_j))` and smooths the sensitivity in `top_opt` per Sigmund
   eq. 5 (fixed in `eaa8e42`; history in B1/B2); `"density"` / `"helmholtz"` filter and
   project the design field itself (explicit weighted `H_f`, or the `src/pde_filter.py`
   PDE solve).
8. **Update** `oc()` - OC with damping 1/2; move limit 0.2 (`"sensitivity"`) or `oc_move`
   (projected path). Volume check is area-weighted `sum(A_e x_e) > volfrac * sum(A_e)`
   (B4 fix), `dv` = per-element areas; the projected path instead passes a `vol_check`
   callback that measures the true volume fraction of the filtered+projected candidate.
9. **Loop** `System.top_opt`: `"sensitivity"` stops on max density change `< change_tol`
   (default 0.01) or `max_iteration` (B3 fix - Sigmund's rule). The projected path
   (`"density"` / `"helmholtz"`) runs beta-continuation; `max_iteration` is the cap, and
   it may stop earlier only on a sustained relative-objective plateau (< 1e-4 over an
   8-iter window) *and* only after `beta_max` has been held `2*beta_iter` (>=20)
   iterations - the raw-design change limit-cycles at the move limit under a sharp
   Heaviside and is not a usable stop signal. `top_opt` prints the stop reason.
10. **Output** `System.plot2()` -> density PDF; then `TrussInputGUI` -> `trusses.json`.

## Mathematical formulation

Minimum-compliance SIMP, 2-D plane stress:
`min_x  c(x) = U^T K(x) U = sum_e x_e^p * u_e^T k0 u_e`
s.t. `K(x) U = F`, `(1/N) sum_e x_e <= volfrac`, `x_min <= x_e <= 1`.
Element: 4-node bilinear quad, 2x2 Gauss, plane stress
`D = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]`, unit thickness.
Filter: selectable (`filter` param) - default Sigmund mesh-independency *sensitivity*
filter (linear hat weights, eq. 5/6; implemented per eq. 5 since `eaa8e42`), or a
volume-weighted *density* filter + tanh projection (see "Filtering options").
Update: Optimality Criteria. References in code: Sigmund (2001) "99 line"
(`reference/sigmund2001.pdf`); DTU 200-line Python code; Xia, Langelaar & Hendriks (2020)
for the downstream STM evaluation.

## Filtering options (`filter` in parameters.json)

The mesh here is an **unstructured gmsh quad mesh with non-uniform element sizes**, so
the `top88`/`top99neo` shortcuts that assume equal unit-area elements do **not** apply
directly. All three filter paths below are written for the general (variable-area) mesh.
`"density"` and `"helmholtz"` share the whole projection / beta-continuation / OC path
in `top_opt` (via a small `forward()`/`chain()` linear-filter adapter); only the linear
filter operator differs.

- **`"sensitivity"` (default)** - Sigmund 2001 eq. 5 sensitivity filter. Design variable
  is the physical density `self.x`. Kept as the parity path. Cheap, robust, but only a
  weak regularizer: leaves grey and can leave isolated hot elements on a fine mesh
  because the `1/x_e` normalization amplifies near-void sensitivities.

- **`"density"`** - volume-weighted linear density filter followed by a smoothed tanh
  threshold projection with beta-continuation. `self.x` then holds the physical
  (filtered+projected) field the FE model sees; `top_opt` tracks the raw design variable
  separately (`self.x_des` after the run).
  - Filter: `x_tilde_e = (sum_i H_ei A_i x_i) / (sum_i H_ei A_i)`, `H_ei = max(0,
    r_min - dist(c_e,c_i))`, `A_i` = element area. This is **Lazarov & Sigmund (2011)
    eq. (1)** - the classical density filter of Bruns & Tortorelli (2001, *CMAME*
    190:3443-3459) / Bourdin (2001, *IJNME* 50:2143-2158) written *with element-volume
    weights*, which is the form that stays consistent (converges to the same continuous
    convolution) on a non-uniform mesh. `top99neo` `ft=2` is the `A_i == 1` special case.
    Reuses the existing dense `convolution_operator()` `H`.
  - Projection: `x_bar = (tanh(b*eta) + tanh(b*(x_tilde-eta))) / (tanh(b*eta) +
    tanh(b*(1-eta)))`, applied per element (no mesh-size assumption), **Wang, Lazarov &
    Sigmund (2011)**, *SMO* 43:767-784. `beta` doubles from `beta` to `beta_max` every
    `beta_iter` iterations / on settling; the projection+continuation recipe follows
    Ferrari & Sigmund (2020) `top99neo` `ft=3` (*SMO* 62:2211-2228).
  - Sensitivity chain (physical -> filtered -> raw): `df/dx_i = A_i * (H @ (df/dx_bar *
    dproj / Hs))_i`, `Hs_e = sum_i H_ei A_i`; same for the volume gradient. `oc()` gets
    a `vol_check` callback that filters+projects the trial design and compares its true
    area-weighted volume fraction to `volfrac` (Andreassen et al. 2011 `top88` inner
    loop, generalized).
  - Deviations from `top99neo`: raw design var is bounded at `x_min` (not 0) because the
    material law is a bare `x^p` with no `E_min` floor, so `x_tilde > 0` is needed to
    keep `K` non-singular; OC move limit `oc_move` defaults to 0.1 (not 0.2) to damp the
    0<->1 element limit-cycle a sharp Heaviside provokes; termination is `max_iteration`
    plus an optional sustained-objective-plateau stop that is gated until `beta_max` has
    been held `2*beta_iter` (>=20) iterations (raw-design change stalls at the move limit
    under a sharp projection and is not a usable stop signal there).
  - Practical: wants a larger `max_iteration` (~150-250) to clear all beta levels;
    `r_min` is an absolute length, so scale it with `mesh_el_size` when refining.

- **`"helmholtz"`** - identical projection / beta / OC path to `"density"`; the linear
  filter is the **PDE (Helmholtz) filter**, Lazarov & Sigmund (2011), *IJNME*
  86:765-781. Solves `x_tilde - rc^2 nabla^2 x_tilde = x` (natural, i.e. homogeneous
  Neumann, BC) on the *same Q4 mesh*, with `rc = r_min / (2 sqrt 3)` so the filter's
  first moment matches the `r_min` hat kernel. Implemented in `src/pde_filter.py`
  (`HelmholtzFilter`): assembles scalar `KF = int rc^2 (grad N)^T grad N` and a
  **row-lumped** mass `ML = diag(int N)` (lumping keeps `S = KF + ML` an M-matrix, so
  `S^-1 >= 0` and the filter preserves non-negativity and the constant field exactly -
  a consistent mass matrix produces small negative overshoots that then blow up the OC
  `sqrt`), plus the element->node operator `T[n,e] = int_e N_n`. Forward
  `x_tilde = A^-1 T^T S^-1 T x`, adjoint `chain(g) = T^T S^-1 T (g / A)`; `S` is
  prefactored once with `splu`. `x_tilde` is clipped to `[0,1]` before projection as a
  cheap guard. Advantages over `"density"`: one sparse SPD solve per call instead of the
  dense `O(N^2)` `convolution_operator()` (no neighbour search, no `NxN` matrix), and no
  artificial pull toward zero at the domain boundary. Combining PDE filter + Heaviside
  projection follows Kawamoto et al. (2011), *SMO* 44:19-24. Same `eta`/`beta`/... knobs
  as `"density"`.

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
  `l2-l1 > 1e-4`; `l2` init `1e9` vs `1e5` - functionally equivalent. Homogeneous
  Dirichlet BCs are applied by solving the free/fixed-partitioned block (`solve_FE_csr`),
  equivalent to eliminating the fixed DOFs; only homogeneous BCs are supported.
- **Not carried over from the paper**: multiple load cases (sec. 4.2), passive/active
  element masks (sec. 4.3 - true mesh holes are used instead), penalty continuation
  (Sigmund also uses fixed `p`).

### Parity mode checklist (when validating a Kratos port against the current code)

Parity is defined for `filter == "sensitivity"` (the default). `filter == "density"` and
`filter == "helmholtz"` are deliberate, non-parity regularization changes (density /
PDE filter + Heaviside projection); don't use them when validating a Sigmund/Kratos
parity port.

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
- **Stiffness assembly**: `K_global_csr` / `solve_FE_csr` is O(N) memory - COO triplets
  -> CSR, free/fixed Dirichlet partition. Benchmarked (in PR #3, against the since-removed
  dense `K_global()` path): identical `u` to ~1e-12, ~5-10x faster solve, ~100x less
  memory at the old ~5-6k-element ceiling (974 MB -> 10 MB at 5.4k elements); 32k elements
  solves in ~60 MB. The remaining O(N^2) is `convolution_operator()` - the dense NxN
  filter matrix used by `filter="sensitivity"` and `filter="density"` (no search-window
  restriction); `filter="helmholtz"` avoids it, so it is the scalable filter choice.
- **Legacy / not on the optimization path** (no need to maintain for topology runs):
  `BeamElement` and the frame/STM methods in `system.py` (`strain_energy_beam_truss`,
  `recover_internal_forces`, `sts`, `plot_deformed_stm_sf*`). (The old `regular_mesh`
  analytic-KE branch in `MembraneElement.k_e_global`, its `regular_mesh` constructor
  flag, the dead `forces_element`, and the `Examples/regular_mesh` case were removed -
  `MembraneElement` now always uses the `QuadPlateMembrane` bilinear KE.)

## Kratos compatibility (research direction)

Goal: keep `geometry.json` + `parameters.json` as the single source of truth while Kratos
replaces the FE solver, and later the whole optimizer, behind a thin adapter. The JSON is
already solver-agnostic; nothing in it is CALFEM-specific. Structure: **one trunk**
(`TopOpt`), the solver a runtime-selectable component (like `filter`), built
on short-lived feature branches - no permanent parallel forks.

### Status

- **Phase 1 - `solver = "native" | "kratos_fe"` - MERGED into `TopOpt`** (PR #4).
  `src/kratos_adapter/fe_export.py` (`.mdpa`/`ProjectParameters.json`/
  `StructuralMaterials.json` + one static analysis) and `fe_solver.py`
  (`KratosFESolver`: persistent `ModelPart`, per-element `YOUNG_MODULUS = E0*x_e^p`,
  `sparse_lu`). Kratos does **only** the FE solve; objective / sensitivity / filter /
  OC stay in-repo (native Q4 KE). `System._solve_fe()` dispatches; `solve_FE_kratos()`
  writes displacements back so downstream code stays solver-agnostic.
  `kratos_fe_parity.py` (file-adapter static solve + `native` vs `kratos_fe` full
  `top_opt`) passes to **~1e-13** on `cantilever1` - machine-precision identical, both
  stages; SIMP hook exact (`compliance(x=.5)/compliance(x=1) == 2^p`). Cost ~1.5-2x
  native (warm), one-time `ModelPart` build 0.1-0.8 s. Kratos is imported lazily; the
  repo runs with Kratos absent.

- **Phase 2 - `optimizer = "native" | "kratos_optapp"` - branch `kratos-optimizer`**
  (stacked; needs rebasing onto post-PR-#4 `TopOpt`). `src/kratos_adapter/opt_export.py`
  emits the current `OptimizationAnalysis` config (`material.simp_control` +
  `linear_strain_energy` objective + `mass` constraint + explicit filter + sigmoidal
  projection + `algorithm_gradient_projection`); `System.top_opt()` delegates and pulls
  the final density into `self.x`. **Topology-level** equivalent to native SIMP+OC on
  `cantilever1` (IoU@0.5 ~= 0.89, compliance within ~11%, volume within ~2%) - *not*
  bit-exact: OC vs gradient-projection, explicit vs sigmoidal projection. `mma` would be
  the right constrained optimizer but `StandardizedNLOPTConstraint` is bugged in this
  Kratos build; `slsqp` holds volume exactly but doesn't scale past ~100 design vars.

### Kratos build on this machine (`~/Kratos/bin/Release`, already on PYTHONPATH)

- **Compiled**: `StructuralMechanicsApplication`, `ConstitutiveLawsApplication`,
  `LinearSolversApplication`, `OptimizationApplication`, `FluidDynamicsApplication`,
  `IgaApplication`, `RomApplication`. Plus the `nlopt` pip package (only needed for
  `optapp_algorithm="mma"`; the default `gradient_projection` needs just `scipy`).
- **NOT compiled**: `TopologyOptimizationApplication` (**3D-only** - registers only
  `SmallDisplacementSIMPElement3D{3,4,8}N`, no 2D element - *and* does not build against
  current Kratos core; ruled out for this 2D project), `ShapeOptimizationApplication`,
  `MeshingApplication`, `MedApplication`.
- Kratos 10.4, compiled for **Python 3.10**; the importing interpreter must match that
  minor version (the `.venv` and system `python3` are both 3.10.12).
- Verified names in `StructuralMechanicsApplication`: `SmallDisplacementElement2D4N`,
  `SmallDisplacementElement2D3N`, `LinearElasticPlaneStress2DLaw`,
  `LinearElasticPlaneStrain2DLaw`, `PointLoadCondition2D1N`, `LineLoadCondition2D2N`.
- OptApp entry is `optimization_analysis.OptimizationAnalysis` (the older
  `optimizer.CreateOptimizer` path is broken in this build); density TO control is
  `controls/material/simp_control.py` (2D-native - drives `DENSITY`/`YOUNG_MODULUS` on
  standard elements via element-specific `Properties`).

### Component mapping (as implemented)

| in-repo stage | phase 1 `kratos_fe` (StructuralMechanicsApplication) | phase 2 `kratos_optapp` (OptimizationApplication) |
|---|---|---|
| mesh from geometry.json | `fe_export._write_mdpa` -> `model.mdpa` (`SmallDisplacementElement2D4N`); `KratosFESolver` builds it in memory | same `.mdpa`; primal uses `use_input_model_part` |
| `MembraneElement` KE | `SmallDisplacementElement2D4N` + `LinearElasticPlaneStress2DLaw` + `THICKNESS=1.0` | same element; `material.simp_control` drives its `DENSITY`/`YOUNG_MODULUS` |
| `K += x^p k_e` | per-element `Properties`, `YOUNG_MODULUS = E0*x_e^p` set each iter, re-solve | `simp_control` `list_of_materials` [void `(0, x_min*E0)`, solid `(1, E0)`] + `young_modulus` sigmoidal projection (`penalty_factor = penalty`) |
| solve / Dirichlet | `ResidualBasedLinearStrategy` + `ResidualBasedBlockBuilderAndSolver` + `sparse_lu`; DOFs fixed via `node.Fix(...)` | primal = `StructuralMechanicsAnalysis` (linear Static, `sparse_lu`); `assign_vector_variable_process` on DISP sub-model-parts |
| point loads | `PointLoadCondition2D1N` + `POINT_LOAD` on the node | `PointLoadCondition2D1N` + `assign_vector_variable_to_conditions_process` |
| line loads | native turns `load_line` into an equal point force per edge node; the adapter wires those, not `LineLoadCondition2D2N` | idem |
| `compliance()` | still native (`x_e^p u_e^T k0 u_e`) - Kratos only returns `u` | `linear_strain_energy_response_function` (objective) |
| volume constraint | still native (area-weighted `oc()` `vol_check`) | `mass_response_function`, `type "<="`, `scaled_ref_value = volfrac * A_total` |
| filter | still native (`filter` param) | OptApp explicit filter (`filter_radius = r_min`) on the density control |
| `oc()` update | still native OC | `algorithm_gradient_projection` (`correction_size = 1.0`); `slsqp` / `mma` selectable |
| `top_opt()` loop | keeps running; only the FE solve is Kratos | replaced by `OptimizationAnalysis(...).Run()`; final element `DENSITY` -> `self.x` |
| `plot2()` + STM GUI | unchanged | unchanged |

### Phased plan

1. **Kratos as FE solver only** (`StructuralMechanicsApplication`) - **done, merged**
   (PR #4). Machine-precision parity with the native solver.
2. **Kratos as the whole optimizer** (`OptimizationApplication`) - **done, branch
   `kratos-optimizer`**. Topology-level parity. `TopologyOptimizationApplication` was the
   original phase-2 target but is 3D-only and does not build - dropped.
3. **Beyond compliance** (`OptimizationApplication`): multi-load-case, stress
   constraints, MMA, robust/projection formulations, eventually 3D. Same input files;
   extends phase 2's control/response/algorithm wiring.

### Adapter caveats (behavioural differences vs. the native code)

- `kratos_fe` is machine-precision identical. `kratos_optapp` is only topology-level
  equivalent (different algorithm + projection); keep `solver="native"` /
  `optimizer="native"` for a Sigmund-parity reference.
- Kratos is imported lazily (only when a `KratosFESolver` / `run_*` is constructed);
  selecting a Kratos backend without Kratos on `PYTHONPATH` raises a clear `ImportError`.
- Holes need no special handling - gmsh meshes them; Kratos only sees the final mesh.
- Do not modify, build, or install into the `~/Kratos` tree unless asked (the `nlopt`
  pip install into `.venv` was explicitly requested).

## Constraints when working here

- Commit / push only when asked. Feature work goes on a short-lived branch off `TopOpt`,
  PR'd back into `TopOpt`.
- Do not modify, build, or install into the `~/Kratos` tree unless asked.
- `results/optimized_structure.pdf` is a transient output slot (rewritten every run);
  leave it out of commits. `results/kratos_*` run dirs are gitignored.
