import numpy as np
from src.membrane import QuadPlateMembrane

class MembraneElement:
    def __init__(self, nodes):
        self.nodes = nodes
        self.dofs = [nodes[0].dofs[0],nodes[0].dofs[1],nodes[1].dofs[0],nodes[1].dofs[1],nodes[2].dofs[0],nodes[2].dofs[1],nodes[3].dofs[0],nodes[3].dofs[1]]
        self.displacements = np.zeros(8)
        self.system_penalty = 0
        self.E = 30000
        self.nu = 0.15
        self.k_e_matrix = None  # This is the cached stiffness matrix
        
    def element_center(self):
        x_coords = [node.coords[0] for node in self.nodes]
        y_coords = [node.coords[1] for node in self.nodes]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        return [x_center, y_center]
    
    
    def element_area(self):
        x_coords = [node.coords[0] for node in self.nodes]
        y_coords = [node.coords[1] for node in self.nodes]
        
        # Ensure the nodes form a closed loop by repeating the first node at the end
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        # Compute the area using the shoelace formula
        area = 0.5 * abs(
            sum(x_coords[i] * y_coords[i+1] - y_coords[i] * x_coords[i+1] for i in range(len(self.nodes)))
        )
        
        return area
        

    def k_e_global(self):
        # 4-node bilinear quad, plane stress, 2x2 Gauss (cached)
        if self.k_e_matrix is None:
            q_e = QuadPlateMembrane(self.nodes, self.E, self.nu)
            self.k_e_matrix = q_e.calculate_elastic_stiffness_matrix()
        return self.k_e_matrix


    def compliance(self,x):
        """ 
        According to equation 1 of Sigmund 2001
        
        Sigmund, Ole. "A 99 line topology optimization code written in Matlab." Structural and multidisciplinary optimization 21.2 (2001): 120-127.
        """
        
        c_e = self.k_e_global()@self.displacements
        c_e = self.displacements@c_e
        c_e = c_e * np.power(x, self.system_penalty)
        return c_e 
    

    def sensitivity_compliance(self,x):
        """ 
        According to equation 4 of Sigmund 2001
        
        Sigmund, Ole. "A 99 line topology optimization code written in Matlab." Structural and multidisciplinary optimization 21.2 (2001): 120-127.
        """
        
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
        
        
    