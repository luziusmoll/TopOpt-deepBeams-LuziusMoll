import numpy as np
from membrane import QuadPlateMembrane

class BeamElement:
    def __init__(self, nodes):
        self.nodes = nodes
        # Each node has 3 DOFs: x, y, and rotation (phi)
        self.dofs = [
            nodes[0].dofs[0], nodes[0].dofs[1], nodes[0].dofs[2],
            nodes[1].dofs[0], nodes[1].dofs[1], nodes[1].dofs[2]
        ]
        self.displacements = np.zeros(6)  # 6 DOFs (3 per node)
        self.system_penalty = 0
        self.E = 30000  # Young's modulus
        self.I = 0.01   # Moment of inertia (beam property)
        self.A = 1    # Cross-sectional area
        self.k_e_matrix = None  # This is the cached stiffness matrix

    def k_e_local(self):
        if self.k_e_matrix is None:
            L = self.calculate_length()
            E = self.E
            I = self.I
            A = self.A

            # Stiffness matrix for a beam element with 3 DOFs per node (x, y, rotation)
            k = E * np.array([
                [ A/L,      0,          0,      -A/L,       0,          0       ],
                [ 0,        12*I/L**3,  6*I/L**2, 0,     -12*I/L**3, 6*I/L**2 ],
                [ 0,        6*I/L**2,   4*I/L,   0,     -6*I/L**2,   2*I/L    ],
                [-A/L,      0,          0,      A/L,        0,          0       ],
                [ 0,       -12*I/L**3, -6*I/L**2, 0,      12*I/L**3, -6*I/L**2 ],
                [ 0,        6*I/L**2,   2*I/L,   0,     -6*I/L**2,   4*I/L    ]
            ])

            self.k_e_matrix = k

        return self.k_e_matrix  # Return the cached stiffness matrix

    def k_e(self):
        # Find angle phi of element based on node coordinates
        node1_coords = self.nodes[0].coords
        node2_coords = self.nodes[1].coords
        delta_x = node2_coords[0] - node1_coords[0]
        delta_y = node2_coords[1] - node1_coords[1]
        L = self.calculate_length()
        phi = np.arctan2(delta_y, delta_x)

        # Define transformation matrix T
        c = np.cos(phi)
        s = np.sin(phi)
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Transform local stiffness matrix to global stiffness matrix
        k_local = self.k_e_local()
        k_global = T.T @ k_local @ T

        self.k_e_matrix = k_global
        return self.k_e_matrix

    def calculate_length(self):
        # Calculate the length of the beam element based on the node coordinates
        node1_coords = self.nodes[0].coords
        node2_coords = self.nodes[1].coords
        length = np.linalg.norm(np.array(node2_coords) - np.array(node1_coords))
        return length

    def forces_element(self, x):
        return self.k_e_global() @ self.displacements

    def compliance(self, x):
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq1"""
        c_e = self.k_e_global() @ self.displacements
        c_e = self.displacements @ c_e
        c_e = c_e * np.power(x, self.system_penalty)
        return c_e

    def sensitivity_compliance(self, x):
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq4"""
        f_e = self.k_e_global() @ self.displacements
        dc_e = self.displacements @ f_e
        return np.multiply(np.multiply(dc_e, (-self.system_penalty)), np.power(x, self.system_penalty - 1.0))
