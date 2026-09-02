# TopOpt for deep beams

This project is a modular Python application designed for finite element analysis and optimization with a graphical user interface (GUI) for geometry input. The application allows users to define geometric shapes, set parameters for optimization, and visualize the results.

## Project Structure

```
TopOpt for deep beams
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── geometryGUI.py
│   ├── parameterGUI.py
│   ├── system.py
│   ├── system_setup.py
│   ├── beam_element.py
│   ├── membrane_element.py
│   ├── membrane.py
│   ├── pde_filter.py
│   ├── node.py
│   ├── mesh.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   └── file_io.py
├── config
│   ├── parameters.json
│   └── geometry.json
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/luziusmoll/TopOpt-deepBeams-LuziusMoll.git
   cd TopOpt for deep beams
   ```

2. Create and activate a virtual environment 
   ```
   python3 -m venv .venv
   source .venv/bin/activate     # macOS / Linux
   .venv\Scripts\activate        # Windows
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage


You can run the application in two ways:

### 1. GUI Mode (default)

Launch the GUI to input geometry and parameters interactively:

```
python src/main.py
```

### 2. Example Folder Mode

Run a predefined example by specifying a folder containing `geometry.json` and `parameters.json`:

```
python src/main.py --example Examples/bridge_1
```

This loads the geometry and parameters from the folder, writes
`results/optimized_structure.pdf` (and a copy `<folder>/optimized_structure.pdf`),
then opens the strut-and-tie GUI.

Add `--no-gui` for a fully headless run (no prompts, no GUI, no `plt.show()` – just the
PDFs). Useful for batch runs / CI / regenerating the example figures:

```
python src/main.py --example Examples/bridge_1 --no-gui
```

## Configuration

Parameters are read from `parameters.json` (dumped from the GUI to `config/parameters.json`,
or placed in an `Examples/<case>/` folder). All values are JSON **strings**.

### Required parameters

| key | meaning |
|---|---|
| `volfrac` | Target volume fraction – the share of the design-domain area allowed to be solid (`0 < volfrac < 1`). |
| `penalty` | SIMP penalization exponent `p` (typically `3`). Higher values push element densities toward 0/1 but worsen convergence. |
| `x_min` | Lower bound on element density (typically `1e-3`). Keeps the stiffness matrix non-singular; there is no separate `E_min` stiffness floor. |
| `r_min` | Filter radius, in the same length units as the geometry. Sets the minimum member/length scale. It is an absolute length, so **scale it with `mesh_el_size`** when you refine the mesh. |
| `Youngs_modulus` | Young's modulus `E` of the solid material. |
| `Poissons_ratio` | Poisson's ratio `nu` of the solid material. |
| `max_iteration` | Hard cap on optimization iterations. |
| `mesh_el_size` | gmsh target element size (`elSizeFactor`), absolute length units. |

### Optional parameters

All optional keys have hard-coded defaults and are read directly by `System.__init__`
(not by `utils/config.py`), so no GUI/schema change is needed to use them.

| key | default | meaning |
|---|---|---|
| `change_tol` | `0.01` | Convergence tolerance: stop when the largest per-iteration density change falls below this (Sigmund 2001, sec. 3.1). |
| `filter` | `"sensitivity"` | Regularization scheme – `"sensitivity"`, `"density"` or `"helmholtz"` (see **Filters** below). |
| `eta` | `0.5` | Projection threshold (`density` / `helmholtz` only): the filtered value that maps to 0.5 after projection. |
| `beta` | `1.0` | Initial projection sharpness (`density` / `helmholtz` only). **`beta: 0` disables the projection** — the physical field is then just the linear filter of the design variable: grey boundaries, but converges cleanly with no OC/Heaviside limit cycle. |
| `beta_max` | `16.0` | Maximum projection sharpness reached by continuation. |
| `beta_iter` | `25` | Iterations between `beta` doublings during continuation. |
| `oc_move` | `0.1` | Optimality-Criteria move limit for the projected path. The `"sensitivity"` path uses the classic `0.2`. |
| `assembly` | `"sparse"` | FE assembly backend. `"sparse"` builds the stiffness matrix directly in compressed-sparse form (`O(N)` memory); `"dense"` is the original path that allocated a full `n_dof × n_dof` array (`O(N²)` — the old ~5–6k-element ceiling), kept only as a reference. Results are identical to ~1e-12. |
| `solver` | `"native"` | FE solver. `"native"` uses the in-repo assembly + `scipy` sparse solve. `"kratos_fe"` uses **Kratos** `StructuralMechanicsApplication` as the FE solver only (per-element `YOUNG_MODULUS = E₀·xᵖ`, sparse LU); the objective, sensitivities, filter and OC update stay in-repo. Requires a Kratos build on `PYTHONPATH`; results match `"native"` to ~1e-13, at ~1.5–2× the runtime. |

### Filters

- **`"sensitivity"`** (default) – Sigmund's mesh-independency *sensitivity* filter: the
  raw sensitivities are smoothed with a linear (hat) weight of radius `r_min`. Cheap and
  robust, but a weak regularizer – it leaves grey transition regions and can leave
  isolated hot elements on a fine mesh.
  Reference: Sigmund, O. (2001), "A 99 line topology optimization code written in
  Matlab", *Structural and Multidisciplinary Optimization* 21:120–127.

- **`"density"`** – the *design field itself* is filtered (volume-weighted linear
  density filter) and then pushed toward 0/1 by a smoothed `tanh` threshold projection,
  with `beta`-continuation from `beta` to `beta_max`. Produces near-black/white designs;
  wants a larger `max_iteration` (~150–250) to clear all `beta` levels.
  References: density filter – Bruns, T.E. & Tortorelli, D.A. (2001), *CMAME*
  190:3443–3459, and Bourdin, B. (2001), *IJNME* 50:2143–2158; the element-volume-weighted
  form used here – Lazarov, B.S. & Sigmund, O. (2011), *IJNME* 86:765–781, eq. (1);
  projection and `beta`-continuation – Wang, F., Lazarov, B.S. & Sigmund, O. (2011),
  *Structural and Multidisciplinary Optimization* 43:767–784, with the practical recipe
  from Ferrari, F. & Sigmund, O. (2020), *Structural and Multidisciplinary Optimization*
  62:2211–2228 ("top99neo", `ft=3`).

- **`"helmholtz"`** – same projection path, but the linear filter is the PDE
  (Helmholtz-type) filter: it solves a screened-Poisson equation on the FE mesh instead
  of the explicit convolution. One sparse symmetric solve per iteration (no neighbour
  search, no dense `N x N` weight matrix), and no artificial pull toward zero at the
  domain boundary – the better choice for large or strongly graded meshes.
  References: Lazarov, B.S. & Sigmund, O. (2011), *IJNME* 86:765–781; PDE filter combined
  with Heaviside projection – Kawamoto, A. et al. (2011), *Structural and
  Multidisciplinary Optimization* 44:19–24.

Only `"sensitivity"` reproduces the Sigmund (2001) reference behaviour; `"density"` and
`"helmholtz"` are deliberate regularization changes.

### Example `parameters.json`

```json
{
  "volfrac": "0.4",
  "penalty": "3",
  "x_min": "0.001",
  "r_min": "0.15",
  "Youngs_modulus": "30000",
  "Poissons_ratio": "0.15",
  "max_iteration": "200",
  "mesh_el_size": "0.05",
  "filter": "helmholtz"
}
```

Geometry and boundary conditions are read from `geometry.json` (dumped from the GUI to
`config/geometry.json`). Surfaces are defined as polygons by giving a list of nodes;
holes in the first surface are added as further polygons. Boundary conditions (loads,
supports) can be defined on the nodes or on the lines of the polygons.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.