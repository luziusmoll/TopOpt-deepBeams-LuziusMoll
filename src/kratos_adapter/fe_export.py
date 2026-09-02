"""
Emit Kratos StructuralMechanicsApplication input files from a `System`, and run
one linear static analysis.

`System` (from `src.system`) is expected to be already meshed and to have its
boundary conditions resolved onto the node objects, i.e. every `Node` carries
`coords`, `fixed = [bool, bool]` and `forces = [fx, fy]` - the state `main.py`
produces after `SystemSetup.apply_boundary_conditions`.

Mapping to Kratos:
  node                -> Kratos node, Id = (index in system.nodes) + 1
  MembraneElement     -> SmallDisplacementElement2D4N
  E, nu, t=1          -> LinearElasticPlaneStress2DLaw properties (plane stress,
                         unit thickness - matches the native QuadPlateMembrane)
  node.fixed          -> assign_vector_variable_process on DISPLACEMENT
  node.forces         -> PointLoadCondition2D1N + assign_vector_variable_to_conditions_process

Only homogeneous Dirichlet BCs and point loads are handled (the native code
applies line loads as an equal point load on every edge node, so `node.forces`
is already the per-node vector).
"""
import json
import os
import numpy as np

_MDPA = "model.mdpa"
_MAT = "StructuralMaterials.json"
_PARAMS = "ProjectParameters.json"
_MODEL_PART = "Structure"
_DOMAIN_SMP = "Parts_domain"       # holds all nodes+elements, for material assignment


def _bc_groups(system):
    """Group node indices (0-based) by Dirichlet pattern and collect loaded nodes."""
    fix_xy, fix_x, fix_y = [], [], []
    loaded = []
    load_vec = None
    for i, n in enumerate(system.nodes):
        fx, fy = bool(n.fixed[0]), bool(n.fixed[1])
        if fx and fy:
            fix_xy.append(i)
        elif fx:
            fix_x.append(i)
        elif fy:
            fix_y.append(i)
        f = np.asarray(n.forces, dtype=float)
        if np.any(f != 0.0):
            loaded.append(i)
            if load_vec is None:
                load_vec = f.copy()
            elif not np.allclose(f, load_vec):
                raise NotImplementedError(
                    "kratos_adapter phase 1 assumes a single load vector; found "
                    f"{f} vs {load_vec}. Per-node loads need element-wise POINT_LOAD.")
    return dict(fix_xy=fix_xy, fix_x=fix_x, fix_y=fix_y,
                loaded=loaded, load_vec=(load_vec if load_vec is not None else np.zeros(2)))


def _write_mdpa(system, path, groups):
    nodes = system.nodes
    elems = system.elements
    idx = {id(n): i for i, n in enumerate(nodes)}
    L = []
    L.append("Begin ModelPartData\nEnd ModelPartData\n")
    L.append("Begin Properties 1\nEnd Properties\n")

    L.append("Begin Nodes")
    for i, n in enumerate(nodes):
        x, y = float(n.coords[0]), float(n.coords[1])
        L.append(f"  {i + 1}  {x:.16g}  {y:.16g}  0.0")
    L.append("End Nodes\n")

    L.append("Begin Elements SmallDisplacementElement2D4N")
    for e_i, e in enumerate(elems):
        ns = " ".join(str(idx[id(nd)] + 1) for nd in e.nodes)
        L.append(f"  {e_i + 1}  1  {ns}")
    L.append("End Elements\n")

    load_cond_ids = []
    if groups["loaded"]:
        L.append("Begin Conditions PointLoadCondition2D1N")
        for c_i, ni in enumerate(groups["loaded"]):
            load_cond_ids.append(c_i + 1)
            L.append(f"  {c_i + 1}  1  {ni + 1}")
        L.append("End Conditions\n")

    def _smp(name, node_ids_0based, elem_all=False, cond_ids=None):
        L.append(f"Begin SubModelPart {name}")
        L.append("  Begin SubModelPartNodes")
        for ni in node_ids_0based:
            L.append(f"    {ni + 1}")
        L.append("  End SubModelPartNodes")
        L.append("  Begin SubModelPartElements")
        if elem_all:
            for e_i in range(len(elems)):
                L.append(f"    {e_i + 1}")
        L.append("  End SubModelPartElements")
        L.append("  Begin SubModelPartConditions")
        for ci in (cond_ids or []):
            L.append(f"    {ci}")
        L.append("  End SubModelPartConditions")
        L.append("End SubModelPart\n")

    _smp(_DOMAIN_SMP, range(len(nodes)), elem_all=True)
    if groups["fix_xy"]:
        _smp("DISP_xy", groups["fix_xy"])
    if groups["fix_x"]:
        _smp("DISP_x", groups["fix_x"])
    if groups["fix_y"]:
        _smp("DISP_y", groups["fix_y"])
    if groups["loaded"]:
        _smp("PointLoad", groups["loaded"], cond_ids=load_cond_ids)

    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def _write_materials(system, path):
    E = float(system.elements[0].E)
    nu = float(system.elements[0].nu)
    mat = {"properties": [{
        "model_part_name": f"{_MODEL_PART}.{_DOMAIN_SMP}",
        "properties_id": 1,
        "Material": {
            "constitutive_law": {"name": "LinearElasticPlaneStress2DLaw"},
            "Variables": {"YOUNG_MODULUS": E, "POISSON_RATIO": nu,
                          "THICKNESS": 1.0, "DENSITY": 1.0},
            "Tables": {},
        },
    }]}
    with open(path, "w", newline="\n") as fh:
        json.dump(mat, fh, indent=4)


def _write_project_parameters(path, out_dir, groups, name):
    constraints = []
    for smp, val in (("DISP_xy", [0.0, 0.0, 0.0]),
                     ("DISP_x", [0.0, None, 0.0]),
                     ("DISP_y", [None, 0.0, 0.0])):
        key = {"DISP_xy": "fix_xy", "DISP_x": "fix_x", "DISP_y": "fix_y"}[smp]
        if groups[key]:
            constraints.append({
                "python_module": "assign_vector_variable_process",
                "kratos_module": "KratosMultiphysics",
                "process_name": "AssignVectorVariableProcess",
                "Parameters": {
                    "model_part_name": f"{_MODEL_PART}.{smp}",
                    "variable_name": "DISPLACEMENT",
                    "value": val,
                    "interval": [0.0, "End"],
                },
            })
    loads = []
    if groups["loaded"]:
        fx, fy = float(groups["load_vec"][0]), float(groups["load_vec"][1])
        loads.append({
            "python_module": "assign_vector_variable_to_conditions_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorVariableToConditionsProcess",
            "Parameters": {
                "model_part_name": f"{_MODEL_PART}.PointLoad",
                "variable_name": "POINT_LOAD",
                "value": [fx, fy, 0.0],
                "interval": [0.0, "End"],
            },
        })

    params = {
        "problem_data": {
            "problem_name": name,
            "parallel_type": "OpenMP",
            "echo_level": 0,
            "start_time": 0.0,
            "end_time": 1.0,
        },
        "solver_settings": {
            "solver_type": "Static",
            "model_part_name": _MODEL_PART,
            "domain_size": 2,
            "echo_level": 0,
            "analysis_type": "linear",
            "model_import_settings": {
                "input_type": "mdpa",
                "input_filename": os.path.join(out_dir, "model"),
            },
            "material_import_settings": {
                "materials_filename": os.path.join(out_dir, _MAT),
            },
            "time_stepping": {"time_step": 1.1},
            "linear_solver_settings": {"solver_type": "skyline_lu_factorization"},
            "rotation_dofs": False,
        },
        "processes": {
            "constraints_process_list": constraints,
            "loads_process_list": loads,
        },
        "output_processes": {},
    }
    with open(path, "w", newline="\n") as fh:
        json.dump(params, fh, indent=4)


def export_static_case(system, out_dir, name="case"):
    """Write model.mdpa, StructuralMaterials.json, ProjectParameters.json into
    out_dir (created if needed). Returns a dict of the three paths."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    groups = _bc_groups(system)
    paths = {
        "mdpa": os.path.join(out_dir, _MDPA),
        "materials": os.path.join(out_dir, _MAT),
        "parameters": os.path.join(out_dir, _PARAMS),
    }
    _write_mdpa(system, paths["mdpa"], groups)
    _write_materials(system, paths["materials"])
    _write_project_parameters(paths["parameters"], out_dir, groups, name)
    return paths


def run_static(out_dir, n_nodes):
    """Run the exported case with Kratos StructuralMechanicsAnalysis and return an
    (n_nodes, 2) array of nodal [ux, uy], indexed by (Kratos node Id - 1)."""
    out_dir = os.path.abspath(out_dir)
    try:
        import KratosMultiphysics as KM
        from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis \
            import StructuralMechanicsAnalysis
    except ImportError as exc:  # pragma: no cover - depends on local Kratos build
        raise ImportError(
            "Kratos is required for run_static. Add ~/Kratos/bin/Release to "
            "PYTHONPATH (see CLAUDE.md 'Kratos build on this machine').") from exc

    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    with open(os.path.join(out_dir, _PARAMS)) as fh:
        params = KM.Parameters(fh.read())

    model = KM.Model()
    StructuralMechanicsAnalysis(model, params).Run()

    mp = model[_MODEL_PART]
    u = np.zeros((n_nodes, 2))
    for nd in mp.Nodes:
        u[nd.Id - 1] = (nd.GetSolutionStepValue(KM.DISPLACEMENT_X),
                        nd.GetSolutionStepValue(KM.DISPLACEMENT_Y))
    return u
