"""
Kratos Multiphysics adapter.

Phase 1 (this module): translate a fully-set-up `System` (mesh + resolved
boundary conditions, exactly as `main.py` builds it) into the standard Kratos
`StructuralMechanicsApplication` input files (`model.mdpa`,
`StructuralMaterials.json`, `ProjectParameters.json`) and run a single linear
static analysis, so the Kratos FE result can be validated against the native
solver. No optimization loop yet.

Kratos is an optional dependency (`~/Kratos/bin/Release` on PYTHONPATH); importing
this package does not import Kratos. `run_static` imports it lazily and raises a
clear error if it is missing.
"""
from src.kratos_adapter.fe_export import export_static_case, run_static
from src.kratos_adapter.fe_solver import KratosFESolver

__all__ = ["export_static_case", "run_static", "KratosFESolver"]
