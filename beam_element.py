import numpy as np

class BeamElement:
    def __init__(self, id, nodes):
        self.nodes = nodes
        self.id = id
        # Each node has 3 DOFs: x, y, and rotation (phi)
        self.dofs = [
            nodes[0].dofs[0], nodes[0].dofs[1], nodes[0].dofs[2],
            nodes[1].dofs[0], nodes[1].dofs[1], nodes[1].dofs[2]
        ]
        self.displacements = np.zeros(6)  # 6 DOFs (3 per node)
        self.system_penalty = 0
        self.E = 30000  # Young's modulus
        self.I = 1   # Moment of inertia (beam property)
        self.A = 1    # Cross-sectional area
        self.k_local = None  # This is the cached stiffness matrix
        self.k_global = None
        

    def k_e_local(self):
        #if self.k_local is None:
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

        return k #self.k_local  # Return the cached stiffness matrix


    def k_e_global(self):
        #if self.k_global is None:
        T = self.Transformationsmatrix()   
    
        # Transform local stiffness matrix to global stiffness matrix
        k_local = self.k_e_local()
        k_global = T.T @ k_local @ T

        #self.k_global = k_global
        return k_global
    
    
    def Transformationsmatrix(self):
    
        # Find angle phi of element based on node coordinates
        node1_coords = self.nodes[0].coords
        node2_coords = self.nodes[1].coords
        delta_x = node2_coords[0] - node1_coords[0]
        delta_y = node2_coords[1] - node1_coords[1]
        L = self.calculate_length()
        #phi = np.arctan2(delta_y, delta_x)

        # Define transformation matrix T
        c = delta_x / L #np.cos(phi)
        s = delta_y / L #np.sin(phi)
        T = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        return T


    def calculate_length(self):
        # Calculate the length of the beam element based on the node coordinates
        node1_coords = self.nodes[0].coords
        node2_coords = self.nodes[1].coords
        length = np.linalg.norm(np.array(node2_coords) - np.array(node1_coords))
        self.L = length
        return length


    def forces_element(self, x):
        return self.k_e_global() @ self.displacements
        

    def Formfunktionen(self, x):
        L = self.L
        xi = 2*x/L-1
        
        N_1 = 0.5*(1-xi)
        N_2 = 1/4* ((1-xi)**2) * (2+xi)
        N_3 = L/8 * ((1-xi)**2) * (1+xi)
        N_4 = 0.5*(1+xi)
        N_5 = 1/4* ((1+xi)**2) * (2-xi)
        N_6 = -L/8 * ((1+xi)**2) * (1-xi)

        N = np.matrix([[N_1,
                        N_2,
                        N_3,
                        N_4,
                        N_5,
                        N_6]])
        
        return N


    def AuswertungFormfunktionen(self, x, d_G):

        N = self.Formfunktionen(x)

        d_L = np.dot(self.Transformationsmatrix(),d_G)
        #homogener Anteil
        u_x_h = d_L[0,0]*N[0,0]+d_L[3,0]*N[0,3]
        v_x_h = d_L[1,0]*N[0,1]+d_L[2,0]*N[0,2]+d_L[4,0]*N[0,4]+d_L[5,0]*N[0,5]

        #partikulärer Anteil
        #v_x_p = (self.q * L**2)/ (24*self.EI) * x**2 * (1 - 2*(x/L) + (x/L)**2)

        #gesamtverschiebung (annahme: keine elementlasten)
        u_x = u_x_h
        v_x = v_x_h
        
        return v_x,u_x
