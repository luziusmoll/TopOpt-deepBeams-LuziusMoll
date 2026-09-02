"""
Parity check: native FE solver vs Kratos StructuralMechanicsApplication.

  1. single static solve via the file adapter (export_static_case + run_static),
     uniform density -> compares displacements + compliance.
  2. a short top_opt run with solver="native" vs solver="kratos_fe" (the
     persistent in-memory KratosFESolver) -> compares objective history and the
     final design.

    python kratos_fe_parity.py [Examples/<case>] [mesh_el_size]

Exit 0 only if both stages pass (rel error < TOL).
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


def build_system(solver="native", uniform=False, el_size=None):
    el_size = EL_SIZE if el_size is None else el_size
    geom = os.path.join(REPO, CASE, "geometry.json")
    param = os.path.join(REPO, CASE, "parameters.json")

    def patched(path):
        c = _orig_lc(path); c["mesh_el_size"] = el_size; return c

    ssmod.load_config = patched
    try:
        ss = SystemSetup(geom)
        nodes, elems = ss.create_mesh_from_geometry()
    finally:
        ssmod.load_config = _orig_lc
    p = _load_config(param)
    p.update({"mesh_el_size": el_size, "solver": solver, "filter": "sensitivity"})
    for e in elems:
        e.E = p["Youngs_modulus"]; e.nu = p["Poissons_ratio"]
    s = System(nodes, elems, p)
    ss.apply_boundary_conditions(s)
    if uniform:
        s.x[:] = 1.0                   # no SIMP scaling -> K == K0, matches Kratos
    return s


def stage1_static():
    s = build_system(uniform=True)
    n_nodes = len(s.nodes)
    print(f"[1] static solve   case={CASE}  el_size={EL_SIZE}  "
          f"nodes={n_nodes}  elements={len(s.elements)}")

    u_native = s.solve_FE_csr().reshape(n_nodes, 2)
    F = s.F_global().reshape(n_nodes, 2)
    c_native = float(np.sum(F * u_native))

    paths = export_static_case(s, OUT, name=os.path.basename(CASE))
    u_kratos = run_static(OUT, n_nodes)
    c_kratos = float(np.sum(F * u_kratos))

    rel_u = np.linalg.norm(u_kratos - u_native) / max(np.linalg.norm(u_native), 1e-300)
    rel_c = abs(c_kratos - c_native) / max(abs(c_native), 1e-300)
    print(f"    files: {', '.join(os.path.relpath(v, REPO) for v in paths.values())}")
    print(f"    compliance native/kratos = {c_native:.10e} / {c_kratos:.10e}")
    print(f"    rel displacement err = {rel_u:.2e}   rel compliance err = {rel_c:.2e}")
    ok = rel_u < TOL and rel_c < TOL
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def stage2_toploop(iters=12):
    print(f"[2] top_opt parity  native vs kratos_fe  ({iters} iters, sensitivity filter)")

    def run(solver):
        s = build_system(solver=solver)
        s.top_opt(s.sensitivity_densitiy(), iters)
        return np.array(s.obj_hist), s.x.copy()

    oh_n, x_n = run("native")
    oh_k, x_k = run("kratos_fe")
    m = min(len(oh_n), len(oh_k))
    rel_obj = (np.abs(oh_n[:m] - oh_k[:m]) / np.maximum(np.abs(oh_n[:m]), 1e-30)).max()
    dx = np.abs(x_n - x_k).max()
    print(f"    iters {len(oh_n)}/{len(oh_k)}   final obj {oh_n[-1]:.10g} / {oh_k[-1]:.10g}")
    print(f"    max rel obj diff = {rel_obj:.2e}   max|dx| = {dx:.2e}")
    ok = rel_obj < TOL and dx < 1e-6
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ok1 = stage1_static()
    print()
    ok2 = stage2_toploop()
    print(f"\n{'ALL PASS' if (ok1 and ok2) else 'FAIL'} (tol {TOL:g})")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
