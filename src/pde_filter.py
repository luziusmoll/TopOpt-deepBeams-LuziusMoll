"""
Helmholtz / PDE density filter for topology optimization.

Solves the screened-Poisson problem

    x_tilde - rc^2 * laplacian(x_tilde) = x        in Omega
    grad(x_tilde) . n = 0                          on dOmega   (natural BC)

on the same Q4 mesh used for the elasticity analysis, instead of the explicit
r_min-radius convolution. Advantages on an unstructured / non-uniform mesh:
no neighbour search, no dense N x N weight matrix (one sparse SPD solve per
call, prefactored once), and no artificial pull toward zero at the domain
boundary.

References
----------
Lazarov, B.S. & Sigmund, O. (2011). "Filters in topology optimization based on
Helmholtz-type differential equations." Int. J. Numer. Methods Eng. 86:765-781.

The length parameter is rc = r_min / (2*sqrt(3)), which matches the first spatial
moment of the classical linear hat filter of radius r_min, so `r_min` keeps the
same meaning as in the explicit filter.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

# 2x2 Gauss rule on the reference square
_GP = np.array([-0.5773502691896257, 0.5773502691896257])
_GW = np.array([1.0, 1.0])


def _shape(xi, eta):
    """Q4 shape functions and their reference-coordinate derivatives.

    Node order matches src/membrane.py: (-1,-1), (+1,-1), (+1,+1), (-1,+1).
    """
    N = 0.25 * np.array([(1 - xi) * (1 - eta),
                         (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta),
                         (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


class HelmholtzFilter:
    """
    Linear density filter x_elem -> x_tilde_elem via the Helmholtz PDE.

    P = diag(1/A) . T^T . S^-1 . T   with   S = KF + MF (symmetric SPD),
    T[n, e] = integral over element e of shape function n, A_e = element area.

    forward(x) applies P; chain(g) applies P^T (for sensitivities, since P is
    not symmetric).
    """

    def __init__(self, elements, nodes, r_min, areas):
        self.areas = np.asarray(areas, dtype=float)
        n_el = len(elements)
        n_nd = len(nodes)
        idx = {id(nd): k for k, nd in enumerate(nodes)}
        rc2 = (r_min / (2.0 * np.sqrt(3.0))) ** 2

        s_rows, s_cols, s_val = [], [], []
        t_rows, t_cols, t_val = [], [], []

        for e_i, e in enumerate(elements):
            xy = np.array([nd.coords for nd in e.nodes], dtype=float)
            enodes = [idx[id(nd)] for nd in e.nodes]

            ke = np.zeros((4, 4))
            me = np.zeros(4)          # row-lumped mass (diagonal)
            fe = np.zeros(4)
            for a, xi in enumerate(_GP):
                for b, eta in enumerate(_GP):
                    N, dN = _shape(xi, eta)
                    J = dN @ xy
                    detJ = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
                    Jinv = np.array([[J[1, 1], -J[0, 1]],
                                     [-J[1, 0], J[0, 0]]]) / (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
                    dNxy = Jinv @ dN
                    w = _GW[a] * _GW[b] * detJ
                    ke += rc2 * (dNxy.T @ dNxy) * w
                    me += N * w        # sum_j N_i N_j lumped -> N_i (partition of unity)
                    fe += N * w

            # Lumped mass keeps S = KF + ML an M-matrix, so S^-1 >= 0 and the
            # filter preserves non-negativity and the constant field exactly
            # (Lazarov & Sigmund 2011 use a lumped/row-summed mass for this).
            se = ke + np.diag(me)
            for p in range(4):
                t_rows.append(enodes[p]); t_cols.append(e_i); t_val.append(fe[p])
                for q in range(4):
                    s_rows.append(enodes[p]); s_cols.append(enodes[q]); s_val.append(se[p, q])

        S = coo_matrix((s_val, (s_rows, s_cols)), shape=(n_nd, n_nd)).tocsc()
        self.T = coo_matrix((t_val, (t_rows, t_cols)), shape=(n_nd, n_el)).tocsc()
        self._lu = splu(S)

    def forward(self, x_elem):
        """Filtered element densities P @ x_elem."""
        x_nd = self._lu.solve(self.T @ np.asarray(x_elem, dtype=float))
        return (self.T.T @ x_nd) / self.areas

    def chain(self, g_elem):
        """P^T @ g_elem: map a d/dx_tilde_elem gradient back to d/dx_elem."""
        y = self._lu.solve(self.T @ (np.asarray(g_elem, dtype=float) / self.areas))
        return self.T.T @ y
