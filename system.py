import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import taichi as ti
from matplotlib import gridspec


class System:
    def __init__(self, nodes, elements, x, r_min=1, volfrac=0.4, penalty=3, x_min=1e-3):
        self.nodes = nodes
        self.elements = elements
        self.penalty = penalty
        self.x = x
        self.x_min = x_min
        self.nr_dofs = nodes[-1].dofs[-1] + 1 ## assumes a continous node numbering !! # nodes[-1].dofs[-1] + 1 
        self.r_min = r_min
        self.volfrac = volfrac

        for e in self.elements:
            e.system_penalty = penalty

     
    def apply_dirichlet_bc(self):
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


    def solve_FE(self):
    
        K_g, F_g = self.return_K_F_dirichlet_bc()
        # print("---> solving FE")
        U =  np.linalg.solve(K_g,F_g)

        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
           
        return U
    

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
    
    
    def solve_FE_taichi(self, num_iterations=1000):
        K_g, F_g = self.return_K_F_dirichlet_bc()
        """
        Solves the FE system K_g * U = F_g using Taichi for GPU-accelerated computations.
        
        Parameters:
        - K_g (np.ndarray): Global stiffness matrix as a dense NumPy array.
        - F_g (np.ndarray): Force vector as a NumPy array.
        - num_iterations (int): Number of iterations for the Jacobi solver.
        
        Returns:
        - U (np.ndarray): Displacement vector as a NumPy array.
        """
        # Initialize Taichi for GPU or CPU, based on availability
        ti.init(arch=ti.gpu)
        
        num_dofs = self.nr_dofs
    
        # Define Taichi fields for K_g, F_g, and U
        K_ti = ti.field(dtype=ti.f32, shape=(num_dofs, num_dofs))  # Stiffness matrix
        F_ti = ti.field(dtype=ti.f32, shape=(num_dofs))            # Force vector
        U_ti = ti.field(dtype=ti.f32, shape=(num_dofs))            # Displacement solution
        
        # Initialize Taichi fields with the provided K_g and F_g arrays
        @ti.kernel
        def initialize_fields(K: ti.types.ndarray(), F: ti.types.ndarray()):
            for i, j in ti.ndrange(num_dofs, num_dofs):
                K_ti[i, j] = K[i, j]
            for i in range(num_dofs):
                F_ti[i] = F[i]
        
        # Run initialization
        initialize_fields(K_g, F_g)
        
        # Jacobi iterative solver
        @ti.kernel
        def jacobi_solver(iterations: int):
            for _ in range(iterations):
                for i in range(num_dofs):
                    sigma = 0.0
                    for j in range(num_dofs):
                        if i != j:
                            sigma += K_ti[i, j] * U_ti[j]
                    U_ti[i] = (F_ti[i] - sigma) / K_ti[i, i]
    
        # Solve using Jacobi iterative solver
        jacobi_solver(num_iterations)
        # Convert solution to a NumPy array and return
        U = U_ti.to_numpy()
        # Assign the computed displacements to elements and nodes
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
    
        return U

    
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
        """ from sigmund2001: A 99 line topology optimization code written in Matlab: eq4"""
        
        dc=[]
        n=0
        for e in self.elements:
            dc.append(e.sensitivity_compliance(self.x[n]))
            n+=1
        return dc
    
    
    def strain_energy_beam_truss(self):
        sum_u_N = 0
        sum_u_B = 0
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
        
        return sum_u_N, sum_u_B


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

    
    def plot2(self, deformed=False, disp_bc=True, line_thickness=0.1, save_path=None):
        """
        Plot elements with the option to save as a PDF with a tight bounding box.
        """
        print("---> plotting elements")
        
        # Setup the colormap
        cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
        norm = Normalize(vmin=0, vmax=1)  # Normalize x from 0 to 1
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    
        # Initialize variables for dynamic axis limits
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
    
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
    
            # Update axis limits dynamically
            x_min, x_max = min(x_min, *xs), max(x_max, *xs)
            y_min, y_max = min(y_min, *ys), max(y_max, *ys)
    
            # Get color based on volume fraction
            color = scalar_map.to_rgba(self.x[n])
    
            # Fill element with appropriate color and outline in black
            plt.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac
            plt.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary in black
            n += 1
    
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
        plt.show()

    
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
                
                
                