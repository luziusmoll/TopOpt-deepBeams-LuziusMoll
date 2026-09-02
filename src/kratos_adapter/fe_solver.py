"""
Persistent in-memory Kratos FE solver for the OC loop.

`KratosFESolver` builds the `ModelPart` once (nodes / SmallDisplacementElement2D4N
/ one Properties per element / DOFs / PointLoad conditions / linear strategy) from
a set-up `System`, then `solve(x, penalty)` per iteration only updates each
element's `YOUNG_MODULUS = E0 * x_e**penalty` and re-solves - no files, no
ModelPart rebuild. This is the SIMP-via-per-element-Properties approach.

Kratos is imported lazily in `__init__`; a clear ImportError is raised if it (or
LinearSolversApplication / StructuralMechanicsApplication) is unavailable.
"""
import numpy as np

_LS_JSON = '{ "solver_type": "sparse_lu" }'


class KratosFESolver:
    def __init__(self, system):
        try:
            import KratosMultiphysics as KM
            import KratosMultiphysics.LinearSolversApplication  # noqa: F401
            import KratosMultiphysics.StructuralMechanicsApplication as SMA
            from KratosMultiphysics import python_linear_solver_factory as plsf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Kratos (with LinearSolversApplication + StructuralMechanicsApplication) "
                "is required for solver='kratos_fe'. Add ~/Kratos/bin/Release to "
                "PYTHONPATH (see CLAUDE.md).") from exc
        self.KM = KM
        POINT_LOAD = SMA.POINT_LOAD           # POINT_LOAD lives in StructuralMechanicsApp
        KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)

        nodes, elems = system.nodes, system.elements
        self.n_nodes = len(nodes)
        self.n_el = len(elems)
        self.E0 = float(elems[0].E)
        nu = float(elems[0].nu)

        self.model = KM.Model()
        mp = self.model.CreateModelPart("Structure")
        mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
        mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
        mp.AddNodalSolutionStepVariable(KM.REACTION)
        mp.AddNodalSolutionStepVariable(POINT_LOAD)

        for i, n in enumerate(nodes):
            mp.CreateNewNode(i + 1, float(n.coords[0]), float(n.coords[1]), 0.0)

        idx = {id(n): i for i, n in enumerate(nodes)}
        law = SMA.LinearElasticPlaneStress2DLaw()
        self.props = []
        for e_i, e in enumerate(elems):
            p = mp.CreateNewProperties(e_i + 1)
            p.SetValue(KM.YOUNG_MODULUS, self.E0)
            p.SetValue(KM.POISSON_RATIO, nu)
            p.SetValue(KM.THICKNESS, 1.0)
            p.SetValue(KM.DENSITY, 1.0)
            p.SetValue(KM.CONSTITUTIVE_LAW, law.Clone())
            conn = [idx[id(nd)] + 1 for nd in e.nodes]
            mp.CreateNewElement("SmallDisplacementElement2D4N", e_i + 1, conn, p)
            self.props.append(p)

        KM.VariableUtils().AddDof(KM.DISPLACEMENT_X, mp)
        KM.VariableUtils().AddDof(KM.DISPLACEMENT_Y, mp)
        for i, n in enumerate(nodes):
            nd = mp.GetNode(i + 1)
            if n.fixed[0]:
                nd.Fix(KM.DISPLACEMENT_X)
            if n.fixed[1]:
                nd.Fix(KM.DISPLACEMENT_Y)

        load_prop = mp.CreateNewProperties(self.n_el + 1)
        c = 0
        for i, n in enumerate(nodes):
            f = np.asarray(n.forces, dtype=float)
            if np.any(f != 0.0):
                c += 1
                mp.CreateNewCondition("PointLoadCondition2D1N", c, [i + 1], load_prop)
                mp.GetNode(i + 1).SetSolutionStepValue(
                    POINT_LOAD, [float(f[0]), float(f[1]), 0.0])

        ls = plsf.ConstructSolver(KM.Parameters(_LS_JSON))
        scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
        bns = KM.ResidualBasedBlockBuilderAndSolver(ls)
        # calc_reactions=False, reform_dofs_each_step=False (keep sparsity),
        # calc_norm_dx=False, move_mesh=False
        self.strategy = KM.ResidualBasedLinearStrategy(mp, scheme, bns,
                                                      False, False, False, False)
        self.strategy.SetEchoLevel(0)
        self.strategy.Initialize()
        self.mp = mp

    def solve(self, x, penalty):
        """Set YOUNG_MODULUS = E0 * x_e**penalty per element, re-solve, return the
        length-(2*n_nodes) displacement vector ordered [ux0, uy0, ux1, uy1, ...]
        (node k -> indices 2k, 2k+1, matching System node.dofs)."""
        KM = self.KM
        E = self.E0 * np.power(np.asarray(x, dtype=float), penalty)
        for p, e_val in zip(self.props, E):
            p.SetValue(KM.YOUNG_MODULUS, float(e_val))
        self.strategy.Solve()
        u = np.empty(2 * self.n_nodes)
        for k in range(self.n_nodes):
            d = self.mp.GetNode(k + 1).GetSolutionStepValue(KM.DISPLACEMENT)
            u[2 * k] = d[0]
            u[2 * k + 1] = d[1]
        return u
