import numpy as np
from membrane import QuadPlateMembrane

class Element:
    def __init__(self, nodes, regular_mesh):
        self.nodes = nodes
        self.dofs = [nodes[0].dofs[0],nodes[0].dofs[1],nodes[1].dofs[0],nodes[1].dofs[1],nodes[2].dofs[0],nodes[2].dofs[1],nodes[3].dofs[0],nodes[3].dofs[1]]
        self.displacements = np.zeros(8)
        self.system_penalty = 0
        self.regular_mesh = regular_mesh
        self.E = 30000
        self.nu = 0.15
        self.k_e_matrix = None  # This is the cached stiffness matrix
        
    def element_center(self):
        x_coords = [node.coords[0] for node in self.nodes]
        y_coords = [node.coords[1] for node in self.nodes]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        return [x_center, y_center]
        

    def k_e_global(self):
        
        if self.regular_mesh == True:
            if self.k_e is None:
                E = self.E
                nu = self.nu
                k = np.array([
                    1.0/2.0-nu/6.0, 1.0/8.0+nu/8.0, -1.0/4.0-nu/12.0, -1.0/8.0+3.0*nu/8.0,
                    -1.0/4.0+nu/12.0, -1.0/8.0-nu/8.0, nu/6.0, 1.0/8.0-3.0*nu/8.0
                ])
        
                self.k_e_matrix = E / (1.0-np.power(nu,2.0)) * np.array([
                    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                    ])
            
        else:
            if self.k_e_matrix is None:  # If stiffness matrix is not yet calculated
                q_e = QuadPlateMembrane(self.nodes, self.E, self.nu)
                self.k_e_matrix = q_e.calculate_elastic_stiffness_matrix()  # Cache result
        return self.k_e_matrix  # Return the cached stiffness matrix

    
    def forces_element(self,x):
        return self.k_e()@self.displacements
    
    def compliance(self,x):
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq1"""
        
        c_e = self.k_e_global()@self.displacements
        c_e = self.displacements@c_e
        c_e = c_e * np.power(x, self.system_penalty)
        return c_e 
    

    def sensitivity_compliance(self,x):
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq4"""
        
        f_e = self.k_e_global()@self.displacements
        dc_e = self.displacements@f_e
        #self.dc = dc_e * (-self.system_penalty) * np.power(x,self.system_penalty-1.0)
        return np.multiply(np.multiply(dc_e, (-self.system_penalty)), np.power(x,self.system_penalty-1.0))
    
    
    def stresses_at_element_center(self):
        q_e = QuadPlateMembrane(self.nodes,self.E,self.nu)
        sigma_e = q_e.recover_stresses_at_center(self.displacements)
        return sigma_e
    
    
    def principal_stresses_at_element_center(self):
        sigma = self.stresses_at_element_center()
        sigma_1 = 0.5 * (sigma[0] + sigma[1]) + np.sqrt(0.25*(sigma[0]-sigma[1])**2 + sigma[2]**2)
        sigma_2 = 0.5 * (sigma[0] + sigma[1]) - np.sqrt(0.25*(sigma[0]-sigma[1])**2 + sigma[2]**2)
        alpha = 0.5 * np.arctan2(2*sigma[2],(sigma[0]-sigma[1]))
        return sigma_1, sigma_2, alpha
        
        
    