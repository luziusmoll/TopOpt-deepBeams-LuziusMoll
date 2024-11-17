import numpy as np
from membrane import QuadPlateMembrane

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
        self.I = 0.01   # Moment of inertia (beam property)
        self.A = 1    # Cross-sectional area
        self.k_local = None  # This is the cached stiffness matrix
        self.k_global = None
        

    def k_e_local(self):
        if self.k_local is None:
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

            self.k_local = k

        return self.k_local  # Return the cached stiffness matrix

    def k_e_global(self):
        if self.k_global is None:
            T = self.Transformationsmatrix()   
    
            # Transform local stiffness matrix to global stiffness matrix
            k_local = self.k_e_local()
            k_global = T.T @ k_local @ T
    
            self.k_global = k_global
        return self.k_global
    
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





    

    def Formfunktionen(self, x):
        L = self.L

        #Formfunktionen
        N_1 = 1-x/L
        N_2 = 1 - 3*(x**2)/(L**2) + 2*(x**3)/(L**3)
        N_3 = -x + 2*(x**2)/L - (x**3)/(L**2)
        N_4 = x/L
        N_5 = 3*(x**2)/(L**2) - 2*(x**3)/(L**3)
        N_6 = (x**2)/L - (x**3)/(L**2)

        N = np.matrix([[N_1,
                        N_2,
                        N_3,
                        N_4,
                        N_5,
                        N_6]])
        return N

    def ErsteAbleitungFormfunktionen(self, x):
        
        L = self.L
        d_N_1 = -1/L
        d_N_2 = -6 * x/(L**2) + 6 * (x**2)/(L**3)
        d_N_3 = -1+ 4 * x/L - 3 * (x**2)/(L**2)
        d_N_4 = 1/L
        d_N_5 = 6 * x/(L**2) - 6 * (x**2)/(L**3)
        d_N_6 = 2 * x/L - 3 * (x**2)/(L**2)

        d_N = np.matrix(  [[d_N_1,
                            d_N_2,
                            d_N_3,
                            d_N_4,
                            d_N_5,
                            d_N_6]])
        return d_N
    
    def ZweiteAbleitungFormfunktionen(self, x):
        
        L = self.L
        dd_N_1 = 0
        dd_N_2 = -6/(L**2) + 12 * x/(L**3)
        dd_N_3 = 4/L - 6 * x/(L**2)
        dd_N_4 = 0
        dd_N_5 = 6/(L**2) - 12 * x/(L**3)
        dd_N_6 = 2/L - 6 * x/(L**2)

        dd_N = np.matrix([[dd_N_1,
                            dd_N_2,
                            dd_N_3,
                            dd_N_4,
                            dd_N_5,
                            dd_N_6]])
        return dd_N
    
    def DritteAbleitungFormfunktionen(self, x):
        
        L = self.L
        ddd_N_1 = 0
        ddd_N_2 = 12 * 1/(L**3) 
        ddd_N_3 = -6 * 1/(L**2)
        ddd_N_4 = 0
        ddd_N_5 = -12 * 1/(L**3)
        ddd_N_6 = -6 * 1/(L**2)

        ddd_N = np.matrix([[ddd_N_1,
                            ddd_N_2,
                            ddd_N_3,
                            ddd_N_4,
                            ddd_N_5,
                            ddd_N_6]])

        return ddd_N
    
    def AuswertungFormfunktionen(self, x, d_G, typ):
        L = self.L

        if typ == "Verformung":
            N = self.Formfunktionen(x)
            d_L = np.dot(self.Transformationsmatrix(),d_G)
            #homogener Anteil
            u_x_h = d_L[0,0]*N[0,0]+d_L[3,0]*N[0,3]
            w_x_h = d_L[1,0]*N[0,1]+d_L[2,0]*N[0,2]+d_L[4,0]*N[0,4]+d_L[5,0]*N[0,5]

            #partikulärer Anteil
            #w_x_p = (self.q * L**2)/ (24*self.EI) * x**2 * (1 - 2*(x/L) + (x/L)**2)

            #gesamtverschiebung
            u_x = u_x_h
            w_x = w_x_h #+ w_x_p
            return w_x,u_x

        elif typ == "Querkraft":
            ddd_N = self.DritteAbleitungFormfunktionen(x)
            d_L = np.dot(self.Transformationsmatrix(),d_G)

            #homogener Anteil
            V_x_h = -self.E*self.I * (d_L[1,0]*ddd_N[0,1]+d_L[2,0]*ddd_N[0,2]+d_L[4,0]*ddd_N[0,4]+d_L[5,0]*ddd_N[0,5])

            #partikulärer Anteil
            #V_x_p = -self.EI * self.q * L**2 /(24*self.EI) * 1/L * (-12 + 24 * (x/L))

            #Gesamtquerkraft
            V_x = V_x_h #+ V_x_p
            return V_x

        elif typ == "Moment":
            dd_N = self.ZweiteAbleitungFormfunktionen(x)

            d_L = np.dot(self.Transformationsmatrix(),d_G)

            #homogener Anteil
            M_x_h = -self.E* self.I * (d_L[1,0]*dd_N[0,1]+d_L[2,0]*dd_N[0,2]+d_L[4,0]*dd_N[0,4]+d_L[5,0]*dd_N[0,5])

            #partikulärer Anteil
            #M_x_p = -self.E* self.I * self.q * L**2 /(24*self.EI) * (2 -12 * x/L + 12 * (x**2/L**2))

            #Gesamtmoment
            M_x = M_x_h #+ M_x_p

            return M_x
        
        elif typ == "Normalkraft":
            d_N = self.ErsteAbleitungFormfunktionen(x)
            
            d_L = np.dot(self.Transformationsmatrix(),d_G)
            
            #homogener Anteil
            N_x_h = self.E* self.A * (d_L[0,0]*d_N[0,0]+d_L[3,0]*d_N[0,3]) 
            
            #partikulärer Anteil
            # N_x_p = 0
            
            #Gesamtnormalkraft
            N_x = N_x_h #+ N_x_p
            
            return N_x
        
        else:
            print("Fehler: Zulässige Eingaben für Auswertung sind: Verformung, Querkraft oder Moment")

