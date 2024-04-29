import numpy as np
import numpy.linalg as la


class QuadPlateMembrane():

    def __init__(self, nodes):
        self.nodes = nodes

    def node_coords(self):
        """array_like: nodal coordinates matrix
        """
        node_coords = np.array( [node.coords for node in self.nodes] )
        return node_coords

    def calculate_elastic_stiffness_matrix(self):

        K_e = np.zeros((8,8))

        GP = np.array([-0.577350269189626, 0.577350269189626])
        WtFac = np.array([1., 1.])

        D = self._comp_mat_matrix_plane_stress()

        for i in range(len(GP)):
            gp_eta = GP[i]         #current gauss point coordinate eta
            gp_w_eta = WtFac[i]    #current gauss point weight in eta
            for j in range(len(GP)):
                gp_xi = GP[j]      #current gauss point coordinate xi
                gp_w_xi = WtFac[j] #current gauss point weight in xi

                # Jacobian (and inverse and determinant)
                J = self._calculate_Jacobian(gp_xi, gp_eta)
                J_inv = la.inv(J)
                det_J = la.det(J)

                ##B-matrix
                B = self._calculate_B_matrix(gp_xi, gp_eta, J_inv)

                ##Sum up element stiffness
                K_e += B.T @ D @ B * det_J * gp_w_xi * gp_w_eta

        ##multiply with thickness
        K_e *= 1.0
        
        return K_e
    
    def calculate_shapefunctions_derivatives(self, xi ,eta):

            if len(self.nodes) != 4:
                raise NotImplementedError('Calculation of bilinear shape functions is only valid for a quadrilateral element with 4 nodes')
            #w.r.t xi
            dN1_dxi = ((-1)*(1-eta))/4  #shapefunction node 1
            dN2_dxi = (( 1)*(1-eta))/4  #shapefunction node 2
            dN3_dxi = (( 1)*(1+eta))/4  #shapefunction node 3
            dN4_dxi = ((-1)*(1+eta))/4  #shapefunction node 4
            #w.r.t eta
            dN1_deta = ((1-xi)*(-1))/4  #shapefunction node 1
            dN2_deta = ((1+xi)*(-1))/4  #shapefunction node 2
            dN3_deta = ((1+xi)*( 1))/4  #shapefunction node 3
            dN4_deta = ((1-xi)*( 1))/4  #shapefunction node 4

            dN = np.array([[dN1_dxi,  dN2_dxi,  dN3_dxi,  dN4_dxi],
                            [dN1_deta, dN2_deta, dN3_deta, dN4_deta]])
            return dN

    def _calculate_Jacobian(self, xi, eta):
        nodal_coordinates = self.node_coords()
        dN_dxi_deta = self.calculate_shapefunctions_derivatives(xi, eta)
        J = np.dot(dN_dxi_deta, nodal_coordinates)
        return J
    
    def _comp_mat_matrix_plane_stress(self):
        E = 1.0
        prxy = 0.3
        D = np.array([ [1,    prxy, 0         ],
                        [prxy, 1,    0         ],
                        [0,    0,    (1-prxy)/2] ])
        D *= E / (1 - prxy**2)
        return D

    
    def _calculate_B_matrix(self, xi, eta, J_inv):

        B = np.zeros((3,8))      #initialize B
        dN_dxi_deta = self.calculate_shapefunctions_derivatives(xi, eta)
        dN_dx_dy = np.dot(J_inv, dN_dxi_deta)
        n_nodes = len(self.nodes)
        for j in range(n_nodes):
            B[0,j*2+0] = dN_dx_dy[0,j]
            B[0,j*2+1] = 0
            B[1,j*2+0] = 0
            B[1,j*2+1] = dN_dx_dy[1,j]
            B[2,j*2+0] = dN_dx_dy[1,j]
            B[2,j*2+1] = dN_dx_dy[0,j]
        return B
    
