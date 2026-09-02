"""
Parity check: native FE solver vs Kratos StructuralMechanicsApplication on the
same mesh / BCs / material (uniform density, no SIMP scaling).

    python kratos_fe_parity.py [Examples/<case>] [mesh_el_size]

Exit 0 on PASS (relative displacement error < TOL), 1 on FAIL.
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import src.system_setup as ssmod
from src.system_setup import SystemSetup
from src.system import System
from src.utils.config import load_config as _load_config
from src.kratos_adapter import export_static_case, run_static

TOL = 1e-6
CASE = sys.argv[1] if len(sys.argv) > 1 else "Examples/cantilever1"
EL_SIZE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
OUT = os.path.join(REPO, "results", "kratos_fe_parity")

_orig_lc = ssmod.load_config


def build_system():
    geom = os.path.join(REPO, CASE, "geometry.json")
    param = os.path.join(REPO, CASE, "parameters.json")

    def patched(path):
        c = _orig_lc(path); c["mesh_el_size"] = EL_SIZE; return c

    ssmod.load_config = patched
    try:
        ss = SystemSetup(geom)
        nodes, elems = ss.create_mesh_from_geometry()
    finally:
        ssmod.load_config = _orig_lc
    p = _load_config(param); p["mesh_el_size"] = EL_SIZE
    for e in elems:
        e.E = p["Youngs_modulus"]; e.nu = p["Poissons_ratio"]
    s = System(nodes, elems, p)
    ss.apply_boundary_conditions(s)
    s.x[:] = 1.0                       # no SIMP scaling -> K == K0, matches Kratos
    return s


def main():
    s = build_system()
    n_nodes = len(s.nodes)
    print(f"case={CASE}  el_size={EL_SIZE}  nodes={n_nodes}  elements={len(s.elements)}")

    u_native = s.solve_FE_csr().reshape(n_nodes, 2)
    F = s.F_global().reshape(n_nodes, 2)
    c_native = float(np.sum(F * u_native))          # F.u == u^T K u == compliance

    paths = export_static_case(s, OUT, name=os.path.basename(CASE))
    print("wrote:", {k: os.path.relpath(v, REPO) for k, v in paths.items()})
    u_kratos = run_static(OUT, n_nodes)
    c_kratos = float(np.sum(F * u_kratos))

    denom = max(np.linalg.norm(u_native), 1e-300)
    rel_u = np.linalg.norm(u_kratos - u_native) / denom
    rel_c = abs(c_kratos - c_native) / max(abs(c_native), 1e-300)
    max_abs = np.abs(u_kratos - u_native).max()

    print(f"\n|u|_native            = {np.linalg.norm(u_native):.6e}")
    print(f"compliance native     = {c_native:.10e}")
    print(f"compliance kratos     = {c_kratos:.10e}   (rel diff {rel_c:.2e})")
    print(f"rel displacement err  = {rel_u:.2e}   max|du| = {max_abs:.2e}")

    ok = rel_u < TOL
    print(f"\n{'PASS' if ok else 'FAIL'} (tol {TOL:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
