import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
#import taichi as ti
from matplotlib import gridspec

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import oc


class System:
    def __init__(self, nodes, elements, parameters):
        self.nodes = nodes
        self.elements = elements
        self.penalty = parameters['penalty'] 
        self.x = np.ones(len(elements)) * parameters['volfrac']
        self.x_min = parameters['x_min'] 
        self.nr_dofs = nodes[-1].dofs[-1] + 1 ## assumes a continous node numbering !! # nodes[-1].dofs[-1] + 1 
        self.r_min = parameters['r_min']
        self.volfrac = parameters['volfrac']
        # Convergence tolerance on the max density change per iteration
        # (Sigmund 2001, sec. 3.1 uses 0.01). Optional override via parameters.json.
        self.change_tol = float(parameters.get('change_tol', 0.01))

        # Regularization / filtering scheme. All optional, defaulted, overridable
        # via parameters.json:
        #   "sensitivity" (default) - Sigmund 2001 eq. 5 sensitivity filter; the
        #       design variable IS the physical density (self.x). Kept for parity.
        #   "density" - volume-weighted linear density filter (Lazarov & Sigmund
        #       2011, IJNME 86:765-781 eq. 1 = Bruns & Tortorelli 2001 / Bourdin
        #       2001 with element-area weights) + smoothed tanh threshold
        #       projection (Wang, Lazarov & Sigmund 2011, SMO 43:767-784) with
        #       beta-continuation after Ferrari & Sigmund (2020) "top99neo" ft=3.
        #   "helmholtz" - same projection path but the linear filter is the PDE
        #       filter of Lazarov & Sigmund (2011): one sparse SPD solve instead
        #       of the dense N x N convolution; better for large / unstructured
        #       meshes and no boundary pull-to-zero.
        #   "density"/"helmholtz": self.x holds the physical (filtered+projected)
        #       field; top_opt tracks the raw design variable (-> self.x_des).
        self.filter = str(parameters.get('filter', 'sensitivity')).lower()
        self.proj_eta = float(parameters.get('eta', 0.5))       # projection threshold
        self.beta0 = float(parameters.get('beta', 1.0))         # initial sharpness
        self.beta_max = float(parameters.get('beta_max', 16.0))
        self.beta_iter = int(parameters.get('beta_iter', 25))   # iters between doublings
        # beta = 0 disables the Heaviside projection entirely: the physical field
        # is then just the linear (density / PDE) filter of the design variable.
        # This loses the sharp 0/1 edges but converges cleanly - no OC+Heaviside
        # limit cycle - which is the better choice for hard load cases.
        self.project = self.beta0 > 0.0
        if self.filter in ('density', 'helmholtz'):
            if self.beta0 < 0.0:
                raise ValueError(f"'beta' must be >= 0 (0 disables projection), got {self.beta0}")
            if self.project:
                # eta at exactly 0 or 1 puts the projection threshold on the
                # boundary: the projected field then cannot reach volfrac and the
                # OC volume bisection diverges (float underflow in oc()).
                if not 0.0 < self.proj_eta < 1.0:
                    raise ValueError(f"projection threshold 'eta' must be in (0, 1), got {self.proj_eta}")
                if self.beta_max < self.beta0:
                    raise ValueError(f"need beta_max >= beta, got beta={self.beta0}, beta_max={self.beta_max}")
                if self.beta_iter < 1:
                    raise ValueError(f"'beta_iter' must be >= 1, got {self.beta_iter}")
        # OC move limit for the density path; a smaller step than the classic 0.2
        # damps the 0<->1 element oscillation the Heaviside projection provokes.
        self.oc_move = float(parameters.get('oc_move', 0.1))

        # FE assembly / solve backend:
        #   "sparse" (default) - K_global_csr() assembles COO triplets straight
        #              into CSR (no dense array), homogeneous Dirichlet by
        #              free/fixed partition, spsolve on the reduced block.
        #              O(N) memory; verified identical to "dense" to ~1e-12.
        #   "dense"  - original path: K_global() builds a dense nr_dofs x nr_dofs
        #              array (O(N^2) memory - the old ~5-6k element ceiling),
        #              then csr_matrix + spsolve. Kept as a reference.
        self.assembly = str(parameters.get('assembly', 'sparse')).lower()

        # FE solver backend:
        #   "native" (default) - the in-repo solver (assembly above + spsolve).
        #   "kratos_fe"         - Kratos StructuralMechanicsApplication as the FE
        #                         solver only (SmallDisplacementElement2D4N,
        #                         per-element YOUNG_MODULUS = E0*x^p, sparse_lu);
        #                         objective/sensitivity/filter/OC stay in-repo.
        #                         Requires Kratos on PYTHONPATH.
        self.solver = str(parameters.get('solver', 'native')).lower()

        for e in self.elements:
            e.system_penalty = parameters['penalty']

     
    def apply_dirichlet_bc(self):
        if not hasattr(self, 'fixed_dofs'):
            self.fixed_dofs = []
            for n in self.nodes:
                for i,fixed in enumerate(n.fixed):
                    if fixed: self.fixed_dofs.append(n.dofs[i])


    def K_global(self):

        K_g = np.zeros((self.nr_dofs,self.nr_dofs))
        
        n=0
        for e in self.elements:
            
            x_p = np.power(self.x[n], self.penalty)
            k = np.multiply(e.k_e_global(), x_p)

            for i, dof_i in enumerate(e.dofs):
                for j, dof_j in enumerate(e.dofs):
                    K_g[dof_i,dof_j] += k[i,j] 
            
            n+=1

        return K_g
    

    def F_global(self):

        F_g = np.zeros(self.nr_dofs)

        for n in self.nodes:
            for i, dof_i in enumerate(n.dofs):
                F_g[dof_i] += n.forces[i]
        
        return F_g
    

    def return_K_F_dirichlet_bc(self):
        
        K_g = self.K_global()
        F_g = self.F_global()
        
        # prescribed displ = 0.0
        for fixed_dof in self.fixed_dofs:
            for dof_i in range(self.nr_dofs):
                K_g[fixed_dof, dof_i] = 0.0
                K_g[dof_i, fixed_dof] = 0.0
                K_g[fixed_dof,fixed_dof] = 1.0

            F_g[fixed_dof] = 0.0

        return K_g, F_g


    def solve_FE_sparse(self):
        K_g, F_g = self.return_K_F_dirichlet_bc()
        # Convert K_g to a sparse matrix format (Compressed Sparse Row format)
        K_g_sparse = csr_matrix(K_g)

        U = spsolve(K_g_sparse, F_g)

        # Assign the computed displacements to elements and nodes
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]

        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]

        return U

    def _solve_fe(self):
        """Dispatch to the configured FE backend."""
        if self.solver == 'kratos_fe':
            return self.solve_FE_kratos()
        if self.assembly == 'sparse':
            return self.solve_FE_csr()
        return self.solve_FE_sparse()

    def solve_FE_kratos(self):
        """FE solve via a persistent Kratos ModelPart (built once, then only
        YOUNG_MODULUS updated + re-solved). Writes displacements back onto the
        node/element objects; downstream compliance()/sensitivity_compliance()
        use the native Q4 KE (verified to agree with Kratos to ~1e-13)."""
        if not hasattr(self, '_kratos_solver'):
            from src.kratos_adapter.fe_solver import KratosFESolver
            self._kratos_solver = KratosFESolver(self)
        U = self._kratos_solver.solve(self.x, self.penalty)
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
        return U

    def _prep_sparse_assembly(self):
        """One-time setup for the sparse assembly path: element->DOF map, the
        per-element 8x8 KE stack, the fixed COO (row, col) index pattern, and
        the free-DOF list for the homogeneous Dirichlet partition."""
        if getattr(self, '_sparse_ready', False):
            return
        n_el = len(self.elements)
        edof = np.asarray([e.dofs for e in self.elements], dtype=np.int64)   # (n_el, 8)
        Ke = np.asarray([e.k_e_global() for e in self.elements], dtype=float)  # (n_el, 8, 8)
        # global (row, col) for every local (i, j) entry of every element
        self._coo_rows = np.repeat(edof, 8, axis=1).ravel()          # edof[e,i] block-repeated
        self._coo_cols = np.tile(edof, (1, 8)).ravel()               # edof[e,j] tiled
        self._edof = edof
        self._Ke_all = Ke
        fixed = np.zeros(self.nr_dofs, dtype=bool)
        if getattr(self, 'fixed_dofs', None):
            fixed[np.asarray(self.fixed_dofs, dtype=np.int64)] = True
        self._free = np.flatnonzero(~fixed)
        self._sparse_ready = True

    def K_global_csr(self):
        """Assemble the global stiffness matrix straight into CSR, no dense
        nr_dofs x nr_dofs array. K = sum_e x_e^p * Ke (Sigmund bare power law)."""
        self._prep_sparse_assembly()
        xp = np.power(self.x, self.penalty)                          # (n_el,)
        data = (self._Ke_all * xp[:, None, None]).ravel()           # (n_el*64,)
        K = coo_matrix((data, (self._coo_rows, self._coo_cols)),
                       shape=(self.nr_dofs, self.nr_dofs)).tocsr()   # sums duplicates
        return K

    def solve_FE_csr(self):
        """FE solve on the CSR matrix with homogeneous Dirichlet applied by
        free/fixed partition (equivalent to zeroing fixed rows/cols)."""
        self._prep_sparse_assembly()
        K = self.K_global_csr()
        F = self.F_global()
        free = self._free
        U = np.zeros(self.nr_dofs)
        U[free] = spsolve(K[free][:, free].tocsc(), F[free])

        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
        return U
    
    
    # def solve_FE_taichi(self, num_iterations=1000):
    #     K_g, F_g = self.return_K_F_dirichlet_bc()
    #     """
    #     Solves the FE system K_g * U = F_g using Taichi for GPU-accelerated computations.
        
    #     Parameters:
    #     - K_g (np.ndarray): Global stiffness matrix as a dense NumPy array.
    #     - F_g (np.ndarray): Force vector as a NumPy array.
    #     - num_iterations (int): Number of iterations for the Jacobi solver.
        
    #     Returns:
    #     - U (np.ndarray): Displacement vector as a NumPy array.
    #     """
    #     # Initialize Taichi for GPU or CPU, based on availability
    #     ti.init(arch=ti.gpu)
        
    #     num_dofs = self.nr_dofs
    
    #     # Define Taichi fields for K_g, F_g, and U
    #     K_ti = ti.field(dtype=ti.f32, shape=(num_dofs, num_dofs))  # Stiffness matrix
    #     F_ti = ti.field(dtype=ti.f32, shape=(num_dofs))            # Force vector
    #     U_ti = ti.field(dtype=ti.f32, shape=(num_dofs))            # Displacement solution
        
    #     # Initialize Taichi fields with the provided K_g and F_g arrays
    #     @ti.kernel
    #     def initialize_fields(K: ti.types.ndarray(), F: ti.types.ndarray()):
    #         for i, j in ti.ndrange(num_dofs, num_dofs):
    #             K_ti[i, j] = K[i, j]
    #         for i in range(num_dofs):
    #             F_ti[i] = F[i]
        
    #     # Run initialization
    #     initialize_fields(K_g, F_g)
        
    #     # Jacobi iterative solver
    #     @ti.kernel
    #     def jacobi_solver(iterations: int):
    #         for _ in range(iterations):
    #             for i in range(num_dofs):
    #                 sigma = 0.0
    #                 for j in range(num_dofs):
    #                     if i != j:
    #                         sigma += K_ti[i, j] * U_ti[j]
    #                 U_ti[i] = (F_ti[i] - sigma) / K_ti[i, i]
    
    #     # Solve using Jacobi iterative solver
    #     jacobi_solver(num_iterations)
    #     # Convert solution to a NumPy array and return
    #     U = U_ti.to_numpy()
    #     # Assign the computed displacements to elements and nodes
    #     for e in self.elements:
    #         for i, dofi in enumerate(e.dofs):
    #             e.displacements[i] = U[dofi]
    
    #     for n in self.nodes:
    #         for i, dofi in enumerate(n.dofs):
    #             n.displacements[i] = U[dofi]
    
    #     return U

    
    def element_centers(self):
        centers = []
        for e in self.elements:
            centers.append(e.element_center())
        return centers
    
    
    def compliance(self):
        sum_c = 0
        n=0
        for e in self.elements:
           sum_c +=  e.compliance(self.x[n]) 
           n+=1
        return sum_c
    
    
    def sensitivity_compliance(self):
        """ 
        According to equation 4 of Sigmund 2001
        
        Sigmund, Ole. "A 99 line topology optimization code written in Matlab." Structural and multidisciplinary optimization 21.2 (2001): 120-127.
        """
        
        dc=[]
        n=0
        for e in self.elements:
            dc.append(e.sensitivity_compliance(self.x[n]))
            n+=1
        return dc

    
    def convolution_operator(self):
        # Convolution operator for mesh independency filtering
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq6"""

        # distance between current element and all others
        element_centers = self.element_centers()
        element_centers = np.array(element_centers)
        
        dist = []
        for i in range(len(self.elements)):
            dist_ij = []
            for j in range(len(self.elements)):
                dist_x = element_centers[i,0]-element_centers[j,0]
                dist_y = element_centers[i,1]-element_centers[j,1]
                dist_ij.append(np.sqrt(dist_x**2 + dist_y**2))
            dist.append(dist_ij)
        
        # convolution operator H_f
        H_f = self.r_min * np.ones([len(element_centers),len(element_centers)]) - dist
        # set negativ values (elements outside of r_min) to zero
        H_f[H_f < 0] = 0
        
        return H_f

    def _project(self, x_tilde, beta):
        """
        Smoothed threshold projection, Wang, Lazarov & Sigmund (2011) eq. 20.
        Maps the filtered field x_tilde in [0, 1] toward 0/1 about self.proj_eta;
        beta controls sharpness (beta -> 0 is the identity).
        """
        eta = self.proj_eta
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / den

    def _dproject(self, x_tilde, beta):
        """d(projected)/d(filtered) for _project()."""
        eta = self.proj_eta
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return beta * (1.0 - np.tanh(beta * (x_tilde - eta)) ** 2) / den

    def top_opt(self, dv, max_iteration):
        # Actual optimization
        """ from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

        loop = 0
        obj_hist = []
        change = 1.0
        v = np.asarray(dv, dtype=float)          # per-element areas = volume weights

        if self.filter not in ('density', 'helmholtz'):
            # ---- sensitivity-filter path (Sigmund 2001 eq. 5), design var == self.x ----
            H = self.convolution_operator()
            x = self.x.copy()
            while change > self.change_tol and loop < max_iteration:
                loop = loop + 1
                self._solve_fe()
                obj = self.compliance()
                obj_hist.append(obj)
                dc = np.asarray(self.sensitivity_compliance(), dtype=float)
                x = self.x.copy()
                # Sigmund 2001 eq. 5 sensitivity filter (B1 fix: divide by sum(H)).
                dc_filtered = np.empty(len(self.elements))
                for i in range(len(self.elements)):
                    if x[i] * np.sum(H[:, i]) > 0:
                        dc_filtered[i] = 1 / (x[i] * np.sum(H[:, i])) * np.sum(H[:, i] * x * dc)
                    else:
                        dc_filtered[i] = dc[i]
                self.x[:] = oc(self.x, self.volfrac, dc_filtered, v, self.x_min)
                change = np.linalg.norm(self.x - x, np.inf)
                if loop % 5 == 0 or loop == 1:
                    print('Iteration:', loop)
                    print('obj:', obj)
                    print('change:', change)
                    print('vol. frac:', np.average(self.x, weights=v))
            self.obj_hist = obj_hist
            return

        # ---- projected density-filter path (top99neo ft=3, generalized) ----
        # self.x carries the physical (filtered+projected) field the FE model
        # sees; xdes is the raw design variable. xdes is bounded at x_min (not 0
        # as in top99neo) because the material law is a bare x^p with no E_min
        # floor, so x_tilde must stay > 0 to keep K non-singular.
        if self.filter == 'helmholtz':
            # PDE filter, Lazarov & Sigmund (2011); no dense N x N weight matrix.
            from src.pde_filter import HelmholtzFilter
            lin = HelmholtzFilter(self.elements, self.nodes, self.r_min, v)
        else:
            # volume-weighted linear hat filter, Lazarov & Sigmund (2011) eq. 1
            H = self.convolution_operator()
            Hs = H @ v

            class _WeightedHat:
                def forward(self_, xe):
                    return (H @ (v * xe)) / Hs

                def chain(self_, ge):
                    return v * (H @ (ge / Hs))

            lin = _WeightedHat()

        Vtot = np.sum(v)
        xdes = self.x.copy()
        project = self.project
        beta = self.beta0
        loop_beta = 0
        loop_at_beta_max = None
        # the objective-plateau stop is only allowed after beta_max has been held
        # this many iterations, so max_iteration stays the primary cap and the
        # design has time to settle after the final sharpening
        beta_max_hold = max(2 * self.beta_iter, 20)
        stop_reason = 'max_iteration'

        def to_phys(xt, b):
            return self._project(np.clip(xt, 0.0, 1.0), b) if project \
                else np.clip(xt, self.x_min, 1.0)

        x_tilde = lin.forward(xdes)
        self.x[:] = to_phys(x_tilde, beta)

        while change > self.change_tol and loop < max_iteration:
            loop = loop + 1
            self._solve_fe()
            obj = self.compliance()
            obj_hist.append(obj)
            dc = np.asarray(self.sensitivity_compliance(), dtype=float)

            # d(physical)/d(filtered): projection slope, or 1 when projection is off
            dpr = np.maximum(self._dproject(x_tilde, beta), 1e-9) if project else 1.0
            # chain rule: physical -> filtered -> raw design variable
            dc_des = lin.chain(dc * dpr)
            dvol_des = np.maximum(lin.chain(v * dpr if project else v), 1e-12)

            def phys_volfrac(xn, _b=beta):
                return np.sum(v * to_phys(lin.forward(xn), _b)) / Vtot

            xdes_old = xdes.copy()
            xdes[:] = oc(xdes, self.volfrac, dc_des, dvol_des, self.x_min,
                         vol_check=phys_volfrac, move=self.oc_move)
            x_tilde = lin.forward(xdes)
            self.x[:] = to_phys(x_tilde, beta)
            # design change on the raw design variable (Sigmund sec. 3.1). With a
            # sharp Heaviside this limit-cycles at the move limit, hence the
            # objective-plateau stop below; without projection it converges normally.
            change = np.linalg.norm(xdes - xdes_old, np.inf)

            if loop % 5 == 0 or loop == 1:
                print('Iteration:', loop)
                print('obj:', obj)
                print('change:', change)
                if project:
                    print('beta:', beta)
                print('vol. frac:', np.sum(v * self.x) / Vtot)

            if project:
                # beta-continuation: sharpen the projection once the design settles
                # or every beta_iter iterations, then keep going.
                if beta < self.beta_max and \
                        (loop_beta + 1 >= self.beta_iter or change <= self.change_tol):
                    beta = min(self.beta_max, 2.0 * beta)
                    loop_beta = 0
                    change = 1.0
                else:
                    loop_beta += 1
                if beta >= self.beta_max and loop_at_beta_max is None:
                    loop_at_beta_max = loop

                # Objective-plateau convergence. The raw design change limit-cycles
                # at the move limit under a sharp Heaviside (grey-band elements
                # flipping), so the objective is the reliable settled signal here.
                # Only checked after beta_max has been held beta_max_hold iterations,
                # so a short max_iteration is never overridden downward by a
                # transient plateau.
                if loop_at_beta_max is not None \
                        and loop - loop_at_beta_max >= beta_max_hold and len(obj_hist) >= 8:
                    w = obj_hist[-8:]
                    rel = abs(w[-1] - np.mean(w[:-1])) / max(abs(w[-1]), 1e-30)
                    if rel < 1e-4:
                        stop_reason = 'objective plateau'
                        change = 0.0

        self.obj_hist = obj_hist
        self.x_des = xdes
        proj_note = f'beta={beta:g}' if project else 'projection=off'
        print(f'top_opt: stopped after {loop} iterations ({stop_reason}); '
              f'obj={obj_hist[-1]:.6g}, vol.frac={np.sum(v * self.x) / Vtot:.4f}, {proj_note}')

    
    
    def strain_energy_beam_truss(self):
        sum_u_N = 0
        sum_u_B = 0
        all_u_N=[]
        all_u_B=[]
        for element in self.elements:
            
            
            # Element displacement vector in the global coordinate system
            u_element_global = np.asarray(element.displacements).reshape(-1, 1)
            
            # Transformation matrix from global to local coordinates
            T = element.Transformationsmatrix()
            
            # Transform displacements to the local coordinate system
            u_element_local = T @ u_element_global
            
            # Element stiffness matrix in the local coordinate system
            k_e_local = np.asarray(element.k_e_local())
            
            # Compute internal force vector in the local coordinate system
            f_e_local = k_e_local @ u_element_local
            
            # strain eneregy normal
            u_N = 0.5*( u_element_local[0] *f_e_local[0] + u_element_local[3] *f_e_local[3])
            
            # strain eneregy bending
            u_B =0.5*( u_element_local[1] *f_e_local[1] + u_element_local[2] *f_e_local[2] + u_element_local[4] *f_e_local[4] + u_element_local[5] *f_e_local[5])
            
            sum_u_N +=  u_N
            sum_u_B += u_B
            
            all_u_N.append(u_N)
            all_u_B.append(u_B)
            
        
        return sum_u_N, sum_u_B #, all_u_N, all_u_B


    def sensitivity_densitiy(self):
        dv = []
        # Element areas:
        for e in self.elements:
            dv.append(e.element_area())
            
        return np.array(dv)
    
  
    def find_and_return_nearest_node(self,search_coords):
        min_dist=10e10
        nearest_node = self.nodes[0]

        for i, n_i in enumerate(self.nodes):

            distance = np.linalg.norm(search_coords-n_i.coords)
            if distance<min_dist:
                min_dist=distance
                nearest_node = n_i

        return nearest_node
    
  
    def fix_node_by_coord(self,fix_coord,fix=[True,True]):
        self.find_and_return_nearest_node(fix_coord).fixed = fix


    def fix_line(self,start_coord,end_coord,fix=[True,True],tol=1e-4):
        
        line = end_coord-start_coord
        length_line = np.linalg.norm(line)

        for n in self.nodes:
            start_to_node = n.coords - start_coord
            length_start_to_node = np.linalg.norm(start_to_node)

            if length_start_to_node<=tol: 
                n.fixed = fix
                continue

            cos_a = line@start_to_node / (length_line * length_start_to_node)
            proj = cos_a * length_start_to_node * line / length_line
            distance = np.linalg.norm(start_to_node - proj)

            if distance <= tol: n.fixed = fix


    def load_line(self,start_coord,end_coord,forces=[0.0,0.0],tol=1e-4):
        
        line = end_coord-start_coord
        length_line = np.linalg.norm(line)
     
        for n in self.nodes:
            start_to_node = n.coords - start_coord
            length_start_to_node = np.linalg.norm(start_to_node)

            if length_start_to_node<=tol: 
                n.forces = forces
                continue

            cos_a = line@start_to_node / (length_line * length_start_to_node)
            proj = cos_a * length_start_to_node * line / length_line
            distance = np.linalg.norm(start_to_node - proj)

            if distance <= tol: n.forces = forces

            
    def load_point(self, load_coord, force=[0.0, 0.0], tol=1e-2):
        
        # Convert load_coord to a numpy array if it isn't already one
        load_coord = np.array(load_coord)
        
        self.find_and_return_nearest_node(load_coord).forces = force

    
    def plot2(self, deformed=False, disp_bc=True, line_thickness=0.1, save_path=None, show=True,
              edges=None):
        """
        Plot the density field as one PolyCollection (fast even for 1e5 elements).
        edges: draw per-element outlines. None -> auto (only for < 2000 elements);
        True/False forces it. Outlines are off for fine meshes so the black mesh
        lines don't swamp the density field.
        Set show=False for headless runs (still writes save_path).
        """
        print("---> plotting elements")
        if edges is None:
            edges = len(self.elements) < 2000

        polys = np.array([[n.current_coords() if deformed else n.coords for n in e.nodes]
                          for e in self.elements], dtype=float)      # (n_el, 4, 2)

        pc = PolyCollection(polys, array=np.asarray(self.x), cmap=plt.cm.gray_r,
                            edgecolors=('black' if edges else 'none'),
                            linewidths=(line_thickness if edges else 0.0), zorder=5)
        pc.set_clim(0.0, 1.0)
        plt.gca().add_collection(pc)

        flat = polys.reshape(-1, 2)
        x_min, y_min = flat.min(axis=0)
        x_max, y_max = flat.max(axis=0)

        # Plot boundary conditions if requested
        if disp_bc:
            print("---> plotting bcs")
            for n in self.nodes:
                if n.fixed[0] or n.fixed[1]:
                    coords = n.current_coords() if deformed else n.coords
                    plt.scatter(coords[0], coords[1], color="red", zorder=10)
                
                if abs(n.forces[0]) > 0 or abs(n.forces[1]) > 0:
                    coords = n.current_coords() if deformed else n.coords
                    plt.scatter(coords[0], coords[1], color="green", zorder=10)
        
        # Set dynamic axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
        # Maintain equal aspect ratio and hide axes
        # plt.axis('equal')
        plt.axis('off')
    
        # Adjust figure margins
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gcf().set_tight_layout(False)
        plt.gcf().set_size_inches((8, 8), forward=True)  # Adjust size as needed
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
        # Save the plot as PDF if save_path is provided
        if save_path:
            print(f"Saving plot to {save_path}")
            plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=150)

        # Display the plot
        if show:
            plt.show()
        else:
            plt.close()

    
    def plot3(self, ax, deformed=False, line_thickness=0.1):
        print("---> plotting elements")

        # Setup the colormap
        cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
        norm = Normalize(vmin=0, vmax=1)  # Normalize x from 0 to 1
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)

        # Start plotting
        n = 0
        for e in self.elements:
            if not deformed:
                coords = [n.coords for n in e.nodes]
            else:
                coords = [n.current_coords() for n in e.nodes]
            
            # Ensure the element is closed by adding the first point at the end
            coords.append(coords[0])
            xs, ys = zip(*coords)

            # Get color based on volume fraction
            color = scalar_map.to_rgba(self.x[n])

            # Fill element with appropriate color and outline in black
            ax.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac
            ax.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary in black
            n += 1

        print("---> plotting bcs")
        for n in self.nodes:
            if n.fixed[0] or n.fixed[1]:
                if deformed == False:
                    ax.scatter([n.coords[0]], [n.coords[1]], color="red", zorder=10)
                else:
                    ax.scatter([n.current_coords()[0]], [n.current_coords()[1]], color="red", zorder=10)
            if deformed == False:
                if abs(n.forces[0]) > 0 or abs(n.forces[1]) > 0:
                    ax.scatter([n.coords[0]], [n.coords[1]], color="green", zorder=10)
            else:
                if abs(n.forces[0]) > 0 or abs(n.forces[1]) > 0:
                    ax.scatter([n.current_coords()[0]], [n.current_coords()[1]], color="green", zorder=10)
        
        ax.grid(True)
        ax.set_aspect('equal')
        
    
    def plot4(self, deformed=False, line_thickness=0.1, disp_bc=True, disp_corner=False):
        print("---> plotting elements")
    
        # Setup the colormap
        cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
        norm = Normalize(vmin=0, vmax=1)  # Normalize x from 0 to 1
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    
        # Create figure and axis
        fig, ax = plt.subplots()
    
        # Initialize the min and max values for xs and ys
        min_xs, min_ys = float('inf'), float('inf')
        max_xs, max_ys = float('-inf'), float('-inf')
        
        for n, e in enumerate(self.elements):  # Use enumerate to track index
            if not deformed:
                coords = [n.coords for n in e.nodes]
            else:
                coords = [n.current_coords() for n in e.nodes]
            
            # Ensure the element is closed by adding the first point at the end
            coords.append(coords[0])
            xs, ys = zip(*coords)
        
            # Update the min and max values for xs and ys
            min_xs = min(min_xs, min(xs))
            max_xs = max(max_xs, max(xs))
            min_ys = min(min_ys, min(ys))
            max_ys = max(max_ys, max(ys))
        
            # Get color based on volume fraction
            color = scalar_map.to_rgba(self.x[n])
        
            # Fill element with appropriate color and outline in black
            ax.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac

        if disp_bc == True:
            print("---> plotting bcs")
            for n in self.nodes:
                if n.fixed[0] or n.fixed[1]:
                    if not deformed:
                        ax.scatter([n.coords[0]], [n.coords[1]], color="red", zorder=10)
                    else:
                        ax.scatter([n.current_coords()[0]], [n.current_coords()[1]], color="red", zorder=10)
        
                if not deformed:
                    if abs(n.forces[0]) > 0 or abs(n.forces[1]) > 0:
                        ax.scatter([n.coords[0]], [n.coords[1]], color="green", zorder=10)
                else:
                    if abs(n.forces[0]) > 0 or abs(n.forces[1]) > 0:
                        ax.scatter([n.current_coords()[0]], [n.current_coords()[1]], color="green", zorder=10)
        if disp_corner == True:           
            # Add a blue dot in the bottom left and top right corners for image processing 
            ax.scatter([min_xs], [min_ys], color="yellow", zorder=10, s=5)  # Bottom left corner
            ax.scatter([max_xs], [max_ys], color="yellow", zorder=10, s=5)  # Top right corner
        
        ax.axis('equal')
        ax.axis('off')  # Turn off the axis
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
    
        # Save the plot as a variable
        plot_variable = fig
    
        plt.close(fig)  # Close the plot to prevent it from displaying in interactive environments
        
        dimensions = [[min_xs, min_ys], [max_xs, max_ys]]
    
        return plot_variable, dimensions


    def combined_plot(self):
        fig = plt.figure(figsize=(18, 5))  # Overall figure size
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Adjust the middle plot width if needed
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Plotting the optimized structure using plot3 method
        self.plot3(ax=ax1, deformed=False)
        ax1.set_title('Mesh Plot')
        ax1.set_aspect('equal')  # Set to 'equal' to maintain original scale (otherwise 'auto')
        
        # Plotting Objective History
        ax2.plot(self.obj_hist)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective')
        ax2.set_title('Objective History')
        ax2.grid(True)
        
        # Plotting the distribution of x
        ax3.hist(self.x, bins=20, alpha=0.75)
        ax3.set_title('Histogram of x')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    

    def recover_internal_forces(self):
        """
        Calculate the internal forces for all elements in the system in the local coordinate system.
    
        Returns:
        - internal_forces: A list of tuples representing the normal, shear, and moment forces for each element.
        """
        internal_forces = []
        for element in self.elements:
            
            # Element displacement vector in the global coordinate system
            u_element_global = np.array(element.displacements).reshape(-1, 1)
            # print(f"Element ID: {element.id}, u_element_global:\n{u_element_global}")
            
            # Transformation matrix from global to local coordinates
            T = element.Transformationsmatrix()
            
            # Transform displacements to the local coordinate system
            u_element_local = T @ u_element_global
            # print(f"Element ID: {element.id}, u_element_local:\n{u_element_local}")
            
            # Compute internal force vector in the local coordinate system
            k_local = element.k_e_local()  # Element stiffness matrix in the local coordinate system
            internal_force_local = k_local @ u_element_local
            
            # Print internal force vector for debugging
            # print(f"Element ID: {element.id}, Internal Force (Local):\n{internal_force_local}")
   
            internal_forces.append(internal_force_local)
        
        return internal_forces

    
    def sts(self):
        """
        Suitable Truss Structure (STS) index according to Xia et al. 2020
        
        Xia, Yi, Matthijs Langelaar, and Max AN Hendriks. "Automated optimization-based generation
        and quantitative evaluation of Strut-and-Tie models." Computers & Structures 238 (2020): 106297.
        """
        internal_forces = self.recover_internal_forces()
        sts = []
        
        # Calculate sts for each element
        for forces in internal_forces:
            s = abs(forces[0]) / (abs(forces[0]) + abs(forces[1]))  # Normalized axial force
            sts.append(s)
        
        # Compute the mean
        return sts


        
    def plot_internal_forces_stm(self, num_points=20):
        """
        Plot the interpolated internal forces (normal, shear, and moment) for all elements in the system.
    
        Parameters:
        - num_points: Number of points along each element for interpolation.
        """
        internal_forces = self.recover_internal_forces()
    
        # Prepare to find global min and max for normalization
        global_normal_values = []
        global_shear_values = []
        global_moment_values = []
    
        for idx, element in enumerate(self.elements):
            internal_force = internal_forces[idx].flatten()  # Flatten to 1D array
    
            # Interpolate forces along the beam
            normal_values = np.linspace(-internal_force[0], internal_force[3], num_points)
            shear_values = np.linspace(-internal_force[1], internal_force[4], num_points)
            moment_values = np.linspace(-internal_force[2], internal_force[5], num_points)
    
            global_normal_values.extend(normal_values)
            global_shear_values.extend(shear_values)
            global_moment_values.extend(moment_values)
    
        # Compute global min and max for normalization
        norm_normal = Normalize(vmin=min(global_normal_values), vmax=max(global_normal_values))
        norm_shear = Normalize(vmin=min(global_shear_values), vmax=max(global_shear_values))
        norm_moment = Normalize(vmin=min(global_moment_values), vmax=max(global_moment_values))
    
        cmap = plt.cm.viridis  # Color map for visualization
        sm_normal = ScalarMappable(norm=norm_normal, cmap=cmap)
        sm_shear = ScalarMappable(norm=norm_shear, cmap=cmap)
        sm_moment = ScalarMappable(norm=norm_moment, cmap=cmap)
    
        # Initialize subplots for normal, shear, and moment forces
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle('Internal Forces (Interpolated) in Elements')
    
        for idx, element in enumerate(self.elements):
            node1_coords = element.nodes[0].coords
            node2_coords = element.nodes[1].coords
            x_coords = np.linspace(node1_coords[0], node2_coords[0], num_points)
            y_coords = np.linspace(node1_coords[1], node2_coords[1], num_points)
    
            # Extract forces
            internal_force = internal_forces[idx].flatten()
    
            # Interpolate forces along the beam
            normal_values = np.linspace(-internal_force[0], internal_force[3], num_points)
            shear_values = np.linspace(-internal_force[1], internal_force[4], num_points)
            moment_values = np.linspace(internal_force[2], internal_force[5], num_points)
    
            # Plot normal forces
            for i in range(num_points - 1):
                axs[0].plot(x_coords[i:i + 2], y_coords[i:i + 2],
                            color=cmap(norm_normal(normal_values[i])), linewidth=2)
            axs[0].set_title('Normal Forces')
            axs[0].set_xlabel('X Coordinate')
            axs[0].set_ylabel('Y Coordinate')
            axs[0].grid(True)
    
            # Plot shear forces
            for i in range(num_points - 1):
                axs[1].plot(x_coords[i:i + 2], y_coords[i:i + 2],
                            color=cmap(norm_shear(shear_values[i])), linewidth=2)
            axs[1].set_title('Shear Forces')
            axs[1].set_xlabel('X Coordinate')
            axs[1].set_ylabel('Y Coordinate')
            axs[1].grid(True)
    
            # Plot moments
            for i in range(num_points - 1):
                axs[2].plot(x_coords[i:i + 2], y_coords[i:i + 2],
                            color=cmap(norm_moment(moment_values[i])), linewidth=2)
            axs[2].set_title('Moments')
            axs[2].set_xlabel('X Coordinate')
            axs[2].set_ylabel('Y Coordinate')
            axs[2].grid(True)
    
        # Add color bars to each subplot
        fig.colorbar(sm_normal, ax=axs[0], orientation='horizontal', label='Normal Force Magnitude')
        fig.colorbar(sm_shear, ax=axs[1], orientation='horizontal', label='Shear Force Magnitude')
        fig.colorbar(sm_moment, ax=axs[2], orientation='horizontal', label='Moment Magnitude')
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

 
    
    def plot_deformed_stm_sf(self, anzahl_auswertepunkte, scale=1.0, title='System Configuration'):
        """
        Plot the undeformed and deformed configuration of the beam system for any arbitrary structure.
    
        Parameters:
        - anzahl_auswertepunkte: Number of evaluation points along each beam for smooth plotting.
        - scale: Scaling factor for the displacements to make deformations visible.
        """
        n = anzahl_auswertepunkte  # Number of points along each beam
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        for element in self.elements:
            # Undeformed beam
            xi, yi = element.nodes[0].coords
            xj, yj = element.nodes[1].coords
            x_undeformed = np.linspace(xi, xj, n)
            y_undeformed = np.linspace(yi, yj, n)
            ax.plot(x_undeformed, y_undeformed, 'k-.', linewidth=1, label='Undeformed' if element == self.elements[0] else "")
    
            # Deformed beam
            D_e_global = element.displacements.reshape((6, 1))  # Global displacement vector
            T = element.Transformationsmatrix()  # Transformation matrix (global to local)
    
            x_deformed, y_deformed = [], []
            L = element.L
            dx_local = L / (n - 1)  # Increment in local coordinates for shape function evaluation
    
            for i in range(n):
                x_e = i * dx_local  # Local x-coordinate along the beam
                v_x, u_x = element.AuswertungFormfunktionen(x_e, D_e_global)  # Local displacements
    
                # Transform local displacements back to global coordinates
                d_e_local = np.matrix([[u_x, v_x, 0, 0, 0, 0]]).T
                d_e_global = T.T @ d_e_local  # Transform to global coordinates
    
                u_x_global = d_e_global[0, 0] * scale
                v_x_global = d_e_global[1, 0] * scale
    
                # Compute global deformed coordinates
                x_global = xi + (xj - xi) * (x_e / L) + u_x_global
                y_global = yi + (yj - yi) * (x_e / L) + v_x_global
                x_deformed.append(x_global)
                y_deformed.append(y_global)
    
            ax.plot(x_deformed, y_deformed, 'b-', linewidth=1.5, label='Deformed' if element == self.elements[0] else "")
    
        # Configure plot
        ax.set_title(f"{title} (scale = {scale})")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')
        plt.show()
    
    
    def plot_deformed_stm_sf_1(self, output_file, dimensions, anzahl_auswertepunkte, scale=1.0):
        """
        Plot the undeformed and deformed configuration of the beam system for any arbitrary structure.
    
        Parameters:
        - anzahl_auswertepunkte: Number of evaluation points along each beam for smooth plotting.
        - scale: Scaling factor for the displacements to make deformations visible.
        """
        n = anzahl_auswertepunkte  # Number of points along each beam
    
        figure_width = 2.87402  # Corresponding to 7.3cm
        figure_height = figure_width  # Maintain aspect ratio
        
        line_width_mm = 1  # Line width in mm
        line_width_points = line_width_mm * 2.8346  # Convert mm to points
        # Set your custom colors (not relevant for binary plotting)
        TUM_blue = (0/255, 101/255, 189/255)
        TUM_red_dark = (217/255, 81/255, 23/255)
        TUM_green = (162/255, 173/255, 0/255)  # Fixed duplicate TUM_blue
        TUM_orange = (227/255, 114/255, 34/255)
        TUM_gray1 = (88/255, 88/255, 90/255)


        # Plot the flipped binary image
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    
        for element in self.elements:
            # Undeformed beam
            xi, yi = element.nodes[0].coords
            xj, yj = element.nodes[1].coords
            x_undeformed = np.linspace(xi, xj, n)
            y_undeformed = np.linspace(yi, yj, n)
            ax.plot(x_undeformed, y_undeformed, color=TUM_gray1, linewidth=line_width_points, label='Undeformed' if element == self.elements[0] else "")
    
            # Deformed beam
            D_e_global = element.displacements.reshape((6, 1))  # Global displacement vector
            T = element.Transformationsmatrix()  # Transformation matrix (global to local)
    
            x_deformed, y_deformed = [], []
            L = element.L
            dx_local = L / (n - 1)  # Increment in local coordinates for shape function evaluation
    
            for i in range(n):
                x_e = i * dx_local  # Local x-coordinate along the beam
                w_x, u_x = element.AuswertungFormfunktionen(x_e, D_e_global)  # Local displacements
                # print(w_x, u_x)
    
                # Transform local displacements back to global coordinates
                d_e_local = np.matrix([[u_x, w_x, 0, 0, 0, 0]]).T
                d_e_global = T.T @ d_e_local  # Transform to global coordinates
    
                u_x_global = d_e_global[0, 0] * scale
                w_x_global = d_e_global[1, 0] * scale
    
                # Compute global deformed coordinates
                x_global = xi + (xj - xi) * (x_e / L) + u_x_global
                y_global = yi + (yj - yi) * (x_e / L) + w_x_global
                x_deformed.append(x_global)
                y_deformed.append(y_global)
    
            ax.plot(x_deformed, y_deformed, color=TUM_blue, linewidth=line_width_points, label='Deformed' if element == self.elements[0] else "")
    
        # Configure plot
        lower_left = dimensions[0]
        upper_right = dimensions[1]
        dx =  (upper_right[0]-lower_left[0])/100
        x_min, x_max = lower_left[0]-dx, upper_right[0]+dx
        y_min, y_max = lower_left[1]-dx, upper_right[1]+dx
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.axis('off')
    
        # Adjust figure margins
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gcf().set_tight_layout(False)
        plt.gcf().set_size_inches((8, 8), forward=True)  # Adjust size as needed
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot as a high-resolution PDF
        output_file = f"{output_file}_scale{scale}.pdf"
        plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0)
        # Display the plot
        plt.show()
        
    
    def delete_short_elements(self, min_dist):
        for e in self.elements:
            # Check if length of any element is going to zero
            if e.calculate_length() < min_dist:
                print(f'Element {e.id} too short')
        
                # Determine which node to delete and which to keep
                id_delete, id_keep = max(e.nodes[0].id, e.nodes[1].id), min(e.nodes[0].id, e.nodes[1].id)
            
        
                # Update element connectivity
                for ele in self.elements:
                    # Update elements connected to the deleted node
                    if ele.nodes[0].id == id_delete:
                        ele.nodes[0] = self.nodes[id_keep]  # Reassign actual Node object
                    if ele.nodes[1].id == id_delete:
                        ele.nodes[1] = self.nodes[id_keep]  # Reassign actual Node object
        
        
                # Remove the node with id_delete
                del self.nodes[id_delete]
        
                # Reassign IDs and DOFs for remaining nodes
                for i in range(len(self.nodes)):
                    self.nodes[i].id = i
                    self.nodes[i].dofs = [3 * i, 3 * i + 1, 3 * i + 2]
                
                # Remove the too-short element
                del self.elements[e.id] 
                
                # Reassign IDs and DOFs for remaining elements
                for i in range(len(self.elements)):
                    self.elements[i].id = i
                    self.elements[i].dofs = [
                        self.elements[i].nodes[0].dofs[0], self.elements[i].nodes[0].dofs[1], self.elements[i].nodes[0].dofs[2],
                        self.elements[i].nodes[1].dofs[0], self.elements[i].nodes[1].dofs[1], self.elements[i].nodes[1].dofs[2]
                    ]
                    
                 
                # Update the number of dofs of the system
                self.nr_dofs = self.nodes[-1].dofs[-1] + 1
                
                
                