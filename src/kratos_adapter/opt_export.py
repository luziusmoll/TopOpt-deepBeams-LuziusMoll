"""
Emit Kratos OptimizationApplication input files for a full density-based
(SIMP) compliance topology optimization, from a set-up `System`, and run it.

Phase 2 of putting Kratos in the loop: Kratos runs the WHOLE optimizer
(objective = compliance, constraint = volume, filter, projection, update), not
just the FE solve. The in-repo OC loop is bypassed.

App note: TopologyOptimizationApplication has no 2D element, so this targets
OptimizationApplication (2D-native: `material.simp_control` drives per-element
DENSITY / YOUNG_MODULUS on standard `SmallDisplacementElement2D4N`).

Files written (into out_dir):
  Structure.mdpa            - mesh + BC sub-model-parts (reused from fe_export)
  StructuralMaterials.json  - LinearElasticPlaneStress2DLaw, initial DENSITY=volfrac
  primal_parameters.json    - StructuralMechanicsAnalysis, use_input_model_part
  optimization_parameters.json - the OptimizationAnalysis config (5 blocks)

`run_optimization(out_dir, n_el)` runs `OptimizationAnalysis(...).Run()` and
returns the final per-element physical density (len n_el, element order).
"""
import json
import os
import numpy as np

from src.kratos_adapter.fe_export import _bc_groups, _write_mdpa, _MODEL_PART, _DOMAIN_SMP

_MAT = "StructuralMaterials.json"
_PRIMAL = "primal_parameters.json"
_OPT = "optimization_parameters.json"
_MDPA_STEM = "Structure"


def _primal_parameters(out_dir, groups):
    constraints = []
    for smp, val in (("DISP_xy", [0.0, 0.0, 0.0]),
                     ("DISP_x", [0.0, None, 0.0]),
                     ("DISP_y", [None, 0.0, 0.0])):
        key = {"DISP_xy": "fix_xy", "DISP_x": "fix_x", "DISP_y": "fix_y"}[smp]
        if groups[key]:
            constraints.append({
                "python_module": "assign_vector_variable_process",
                "kratos_module": "KratosMultiphysics",
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
            "Parameters": {
                "model_part_name": f"{_MODEL_PART}.PointLoad",
                "variable_name": "POINT_LOAD",
                "value": [fx, fy, 0.0],
                "interval": [0.0, "End"],
            },
        })
    params = {
        "problem_data": {"problem_name": "primal", "parallel_type": "OpenMP",
                         "echo_level": 0, "start_time": 0.0, "end_time": 1.0},
        "solver_settings": {
            "solver_type": "Static",
            "model_part_name": _MODEL_PART,
            "domain_size": 2,
            "echo_level": 0,
            "analysis_type": "linear",
            "model_import_settings": {"input_type": "use_input_model_part"},
            "material_import_settings": {"materials_filename": os.path.join(out_dir, _MAT)},
            "time_stepping": {"time_step": 1.0},
            "linear_solver_settings": {"solver_type": "sparse_lu"},
            "rotation_dofs": False,
        },
        "processes": {
            "constraints_process_list": constraints,
            "loads_process_list": loads,
        },
        "output_processes": {},
    }
    with open(os.path.join(out_dir, _PRIMAL), "w", newline="\n") as fh:
        json.dump(params, fh, indent=4)


def _materials(out_dir, E0, nu, volfrac):
    mat = {"properties": [{
        "model_part_name": f"{_MODEL_PART}.{_DOMAIN_SMP}",
        "properties_id": 1,
        "Material": {
            "constitutive_law": {"name": "LinearElasticPlaneStress2DLaw"},
            "Variables": {"YOUNG_MODULUS": E0, "POISSON_RATIO": nu,
                          "THICKNESS": 1.0, "DENSITY": volfrac},
            "Tables": {},
        },
    }]}
    with open(os.path.join(out_dir, _MAT), "w", newline="\n") as fh:
        json.dump(mat, fh, indent=4)


_OBJ = {"response_name": "strain_energy", "type": "minimization", "scaling": 1.0}


def _algorithm_settings(algorithm, max_iter, mass_ub):
    """mass_ub = the absolute mass (= sum rho_e * A_e) upper bound = volfrac * A_total."""
    if algorithm == "slsqp":
        # SciPy SLSQP - enforces the volume constraint exactly, but is a dense
        # QP method: it does NOT scale past ~100 design variables (with thousands
        # of elements it barely moves from x0 and returns a uniform field). Kept
        # for small problems / reference only. gradient_projection is the default.
        return {
            "type": "SciPy_algorithms",
            "SciPy_settings": {
                "method": "SLSQP",
                "lower_bound": 0.0, "upper_bound": 1.0,   # control (phi) bounds
                "options": {"disp": False, "maxiter": int(max_iter)},
            },
            "controls": ["density"],
            "objective": _OBJ,
            "constraints": [{"response_expression": "mass", "upper_boundary": mass_ub}],
        }
    if algorithm == "mma":
        return {
            "type": "NLOPT_algorithms",
            "NLOPT_settings": {
                "algorithm_name": "mma",
                "controls_lower_bound": "0", "controls_upper_bound": "1",
                "stopping_criteria": {"maximum_function_evalualtion": int(max_iter)},
                "algorithm_specific_settings": {"inner_maxeval": 20},
            },
            "controls": ["density"],
            "objective": _OBJ,
            "constraints": [{"response_name": "mass", "type": "<", "ref_value": mass_ub}],
        }
    return {   # gradient_projection
        "type": "algorithm_gradient_projection",
        "settings": {
            "echo_level": 0,
            "line_search": {"type": "const_step", "init_step": 2e-2,
                            "gradient_scaling": "inf_norm"},
            "conv_settings": {"max_iter": int(max_iter),
                              "constraint_conv_settings": "none"},
            "linear_solver_settings": {
                "solver_type": "LinearSolversApplication.dense_col_piv_householder_qr"},
            "correction_size": 1.0,   # cap on the per-iter constraint-restoration step
        },
        "controls": ["density"],
        "objective": _OBJ,
        # StandardizedConstraint wants "scaled_ref_value" (float or "initial_value")
        "constraints": [{"response_name": "mass", "type": "<=",
                         "scaling": 1.0, "scaled_ref_value": mass_ub}],
    }


def _optimization_parameters(out_dir, *, E0, x_min, volfrac, penalty, r_min,
                             max_iter, beta, beta_max, beta_iter, algorithm, total_area):
    design_mp = f"{_MODEL_PART}.{_DOMAIN_SMP}"
    cfg = {
        "problem_data": {"parallel_type": "OpenMP", "echo_level": 0},
        "model_parts": [{
            "settings": {"model_part_name": _MODEL_PART, "domain_size": 2,
                         "input_filename": os.path.join(out_dir, _MDPA_STEM)},
        }],
        "analyses": [{
            "name": "primal",
            "type": "kratos_analysis_execution_policy",
            "settings": {
                "model_part_names": [_MODEL_PART],
                "analysis_module": "KratosMultiphysics.StructuralMechanicsApplication",
                "analysis_type": "StructuralMechanicsAnalysis",
                "analysis_settings": {"@include_json": os.path.join(out_dir, _PRIMAL)},
                "analysis_output_settings": {
                    "nodal_solution_step_data_variables": ["DISPLACEMENT"],
                },
            },
        }],
        "responses": [
            {"name": "mass", "type": "mass_response_function",
             "settings": {"evaluated_model_part_names": [design_mp]}},
            {"name": "strain_energy", "type": "linear_strain_energy_response_function",
             "settings": {"evaluated_model_part_names": [design_mp],
                          "primal_analysis_name": "primal", "perturbation_size": 1e-8}},
        ],
        "controls": [{
            "name": "density",
            "type": "material.simp_control",
            "settings": {
                "controlled_model_part_names": [design_mp],
                "output_all_fields": False,
                "list_of_materials": [
                    {"density": 0.0, "young_modulus": x_min * E0},
                    {"density": 1.0, "young_modulus": E0},
                ],
                "filter_settings": {
                    "filter_type": "explicit_filter",
                    "filter_function_type": "linear",
                    "max_items_in_bucket": 10,
                    "echo_level": 0,
                    "filter_radius_settings": {"filter_radius_type": "constant",
                                              "filter_radius": r_min},
                    "filtering_boundary_conditions": {
                        "damping_type": "nearest_entity",
                        "damping_function_type": "cosine",
                        "damped_model_part_settings": {},
                    },
                },
                "density_projection_settings": {
                    "type": "adaptive_sigmoidal_projection",
                    "initial_value": beta, "max_value": beta_max,
                    "increase_fac": 1.05, "update_period": beta_iter,
                },
                "young_modulus_projection_settings": {
                    "type": "adaptive_sigmoidal_projection",
                    "initial_value": beta, "max_value": beta_max,
                    "increase_fac": 1.05, "update_period": beta_iter,
                    "penalty_factor": penalty,
                },
            },
        }],
        "algorithm_settings": _algorithm_settings(algorithm, max_iter,
                                                  volfrac * total_area),
        "processes": {
            "kratos_processes": {},
            "optimization_data_processes": {
                "output_processes": [{
                    "type": "optimization_problem_ascii_output_process",
                    "module": "KratosMultiphysics.OptimizationApplication.processes",
                    "settings": {
                        "output_file_name": os.path.join(out_dir, "summary.csv"),
                        "write_kratos_version": False, "write_time_stamp": False,
                        "list_of_output_components": ["response_function.mass",
                                                     "response_function.strain_energy"],
                    },
                }],
            },
        },
    }
    with open(os.path.join(out_dir, _OPT), "w", newline="\n") as fh:
        json.dump(cfg, fh, indent=4)


def export_optimization_case(system, out_dir, *, volfrac=None, penalty=None,
                             r_min=None, max_iter=None,
                             beta=8.0, beta_max=32.0, beta_iter=20,
                             algorithm="gradient_projection"):
    """Write the four OptimizationApplication input files. Missing volfrac /
    penalty / r_min / max_iter default to the System's values.
    algorithm: "gradient_projection" (default; scales, volume held to ~2%),
    "slsqp" (exact volume but does not scale) or "mma" (blocked - upstream bug)."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    E0 = float(system.elements[0].E)
    nu = float(system.elements[0].nu)
    volfrac = float(system.volfrac if volfrac is None else volfrac)
    penalty = float(system.penalty if penalty is None else penalty)
    r_min = float(system.r_min if r_min is None else r_min)
    max_iter = int(200 if max_iter is None else max_iter)
    x_min = float(system.x_min)

    groups = _bc_groups(system)
    _write_mdpa(system, os.path.join(out_dir, _MDPA_STEM + ".mdpa"), groups)
    _materials(out_dir, E0, nu, volfrac)
    _primal_parameters(out_dir, groups)
    total_area = float(np.sum([e.element_area() for e in system.elements]))
    _optimization_parameters(out_dir, E0=E0, x_min=x_min, volfrac=volfrac,
                             penalty=penalty, r_min=r_min, max_iter=max_iter,
                             beta=beta, beta_max=beta_max, beta_iter=beta_iter,
                             algorithm=algorithm, total_area=total_area)
    return {"mdpa": os.path.join(out_dir, _MDPA_STEM + ".mdpa"),
            "materials": os.path.join(out_dir, _MAT),
            "primal": os.path.join(out_dir, _PRIMAL),
            "optimization": os.path.join(out_dir, _OPT)}


def run_optimization(out_dir, n_el):
    """Run OptimizationAnalysis on the exported case; return the final
    per-element physical density (len n_el, in mdpa element-id order)."""
    out_dir = os.path.abspath(out_dir)
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.LinearSolversApplication  # noqa: F401
        import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
        import KratosMultiphysics.OptimizationApplication  # noqa: F401
        from KratosMultiphysics.OptimizationApplication.optimization_analysis import OptimizationAnalysis
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Kratos OptimizationApplication is required for run_optimization. "
            "Compile it and add ~/Kratos/bin/Release to PYTHONPATH.") from exc

    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        with open(_OPT) as fh:
            params = KM.Parameters(fh.read())
        model = KM.Model()
        OptimizationAnalysis(model, params).Run()
        mp = model[_MODEL_PART]
        x = np.zeros(n_el)
        for el in mp.Elements:
            x[el.Id - 1] = el.Properties[KM.DENSITY]
        return x
    finally:
        os.chdir(cwd)
