import numpy as np
from membrane import QuadPlateMembrane

class Element:
    def __init__(self, nodes):
        self.nodes = nodes
        self.dofs = [nodes[0].dofs[0],nodes[0].dofs[1],nodes[1].dofs[0],nodes[1].dofs[1],nodes[2].dofs[0],nodes[2].dofs[1],nodes[3].dofs[0],nodes[3].dofs[1]]
        self.displacements = np.zeros(8)
        self.system_penalty = 0
        #self.dc = 0.0
        
    def element_center(self):
        x_coords = [node.coords[0] for node in self.nodes]
        y_coords = [node.coords[1] for node in self.nodes]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        return [x_center, y_center]
        

    def k_e(self):
        
        # for regular/easy mesh 
        E = 1.0
        nu = 0.3
        k = np.array([
            1.0/2.0-nu/6.0, 1.0/8.0+nu/8.0, -1.0/4.0-nu/12.0, -1.0/8.0+3.0*nu/8.0,
            -1.0/4.0+nu/12.0, -1.0/8.0-nu/8.0, nu/6.0, 1.0/8.0-3.0*nu/8.0
        ])

        k_e = E / (1.0-np.power(nu,2.0)) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ])
        
        #q_e = QuadPlateMembrane(self.nodes)
        #k_e = q_e.calculate_elastic_stiffness_matrix()
        
        return k_e
    
    def forces_element(self,x):
        return self.k_e()@self.displacements
    
    def compliance(self,x):
        """
        from sigmund2001
        A 99 line topology optimization code written in Matlab
        eq1
        """
        c_e = self.k_e()@self.displacements
        c_e = self.displacements@c_e
        c_e = c_e * np.power(x, self.system_penalty)
        return c_e 
    

    def sensitivity_compliance(self,x):
        """
        from sigmund2001
        A 99 line topology optimization code written in Matlab
        eq4 
        """
        dc_e = self.k_e()@self.displacements
        dc_e = self.displacements@dc_e
        #self.dc = dc_e * (-self.system_penalty) * np.power(x,self.system_penalty-1.0)
        return np.multiply(np.multiply(dc_e, (-self.system_penalty)), np.power(x,self.system_penalty-1.0))
    