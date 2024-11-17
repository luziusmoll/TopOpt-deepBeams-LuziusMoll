import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
import taichi as ti
import numpy as np
import time



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

        #self.apply_dirichlet_bc()


    def apply_dirichlet_bc(self):
        self.fixed_dofs = []
        for n in self.nodes:
            for i,fixed in enumerate(n.fixed):
                if fixed: self.fixed_dofs.append(n.dofs[i])


    def K_global(self):

        K_g = np.zeros((self.nr_dofs,self.nr_dofs))
        
        n=0
        for e in self.elements:
            
            # if self.x[n] < self.x_min:
            #     x_p = np.power(self.x_min, self.penalty) 
            # else:
            #     x_p = np.power(self.x[n], self.penalty)
            
            x_p = np.power(self.x[n], self.penalty)
            k = np.multiply(e.k_e(), x_p)

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
        
        # print("---> creating Tensors")
        start_time = time.time()
        K_g = self.K_global()
        end_time = time.time()
        #print(f"K_global computation time: {end_time - start_time:.6f} seconds")
        start_time = time.time()
        F_g = self.F_global()
        end_time = time.time()
        #print(f"F_global computation time: {end_time - start_time:.6f} seconds")
      
        start_time = time.time()
        # prescribed displ = 0.0
        for fixed_dof in self.fixed_dofs:
            for dof_i in range(self.nr_dofs):
                K_g[fixed_dof, dof_i] = 0.0
                K_g[dof_i, fixed_dof] = 0.0
                K_g[fixed_dof,fixed_dof] = 1.0

            F_g[fixed_dof] = 0.0

        end_time = time.time()
        #print(f"Applying dirichlet BC computation time: {end_time - start_time:.6f} seconds")
        return K_g, F_g


    def solve_FE(self):
    
        K_g, F_g = self.return_K_F_dirichlet_bc()
        start_time = time.time()
        # print("---> solving FE")
        U =  np.linalg.solve(K_g,F_g)
    
        end_time = time.time()
        print(f"Actual solver computation time: {end_time - start_time:.6f} seconds")
        
        # Assign the computed displacements to elements and nodes
        start_time = time.time()
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
    
        end_time = time.time()
        print(f"Assigning displacements to elements computation time: {end_time - start_time:.6f} seconds")
       
        return U
    


    def solve_FE_sparse(self):
        K_g, F_g = self.return_K_F_dirichlet_bc()
        start_time = time.time()
        # Convert K_g to a sparse matrix format (Compressed Sparse Row format)
        K_g_sparse = csr_matrix(K_g)
    
        #print("---> solving FE using sparse matrix solver")
        
        U = spsolve(K_g_sparse, F_g)
    
        end_time = time.time()
        #print(f"Actual solver computation time: {end_time - start_time:.6f} seconds")
        # Assign the computed displacements to elements and nodes
        start_time = time.time()
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
    
        end_time = time.time()
        #print(f"Assigning displacements to elements computation time: {end_time - start_time:.6f} seconds")
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
        start_time = time.time()
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
        end_time = time.time()
        print(f"Initialize Taichi time: {end_time - start_time:.6f} seconds")
        
        # Jacobi iterative solver
        start_time = time.time()
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
        end_time = time.time()
        print(f"Actual solver computation time: {end_time - start_time:.6f} seconds")
        # Convert solution to a NumPy array and return
        start_time = time.time()
        U = U_ti.to_numpy()
        end_time = time.time()
        print(f"Convert solution to a NumPy array time: {end_time - start_time:.6f} seconds")
        # Assign the computed displacements to elements and nodes
        start_time = time.time()
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
    
        end_time = time.time()
        print(f"Assigning displacements to elements computation time: {end_time - start_time:.6f} seconds")
    
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


    def plot(self, deformed=False, line_thickness=0.2):
        
        print("---> plotting elements")
        for e in self.elements:
            coord = []

            if deformed==False:
                for n in e.nodes:
                    coord.append([n.coords[0],n.coords[1]])
                coord.append([e.nodes[0].coords[0],e.nodes[0].coords[1]])
            else:
                for n in e.nodes:
                    coord.append([n.current_coords()[0],n.current_coords()[1]])
                coord.append([e.nodes[0].current_coords()[0],e.nodes[0].current_coords()[1]])

            xs, ys = zip(*coord) #create lists of x and y values
            plt.fill(xs,ys,color="lightgrey",zorder=5)
            plt.plot(xs,ys,color="black",zorder=6, linewidth=line_thickness)

        print("---> plotting bcs")
        for n in self.nodes:
            if n.fixed[0] or n.fixed[1]:
                plt.scatter([n.current_coords()[0]],[n.current_coords()[1]],color="red",zorder=10)
                        
            if deformed == False:
                if abs(n.forces[0])>0 or abs(n.forces[1])>0:
                    plt.scatter([n.coords[0]],[n.coords[1]],color="green",zorder=10)
            else:
                if abs(n.forces[0])>0 or abs(n.forces[1])>0:
                    plt.scatter([n.current_coords()[0]],[n.current_coords()[1]],color="green",zorder=10)
            
        plt.grid()
        plt.axis('equal')
        plt.show()
        
        
    def plot2(self, deformed=False,line_thickness=0.1):
        print("---> plotting elements")

        # Setup the colormap
        cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
        norm = Normalize(vmin=0, vmax=1)  # Normalize x from 0 to 1
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)

        # Start plotting
        n = 0
        for e in self.elements:
            #coord = []
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
            plt.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac
            plt.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary in black
            n+=1

        print("---> plotting bcs")
        for n in self.nodes:
            if n.fixed[0] or n.fixed[1]:
                if deformed == False:
                    plt.scatter([n.coords[0]],[n.coords[1]],color="red",zorder=10)
                else:
                    plt.scatter([n.current_coords()[0]],[n.current_coords()[1]],color="red",zorder=10)
            
            
            if deformed == False:
                if abs(n.forces[0])>0 or abs(n.forces[1])>0:
                    plt.scatter([n.coords[0]],[n.coords[1]],color="green",zorder=10)
            else:
                if abs(n.forces[0])>0 or abs(n.forces[1])>0:
                    plt.scatter([n.current_coords()[0]],[n.current_coords()[1]],color="green",zorder=10)
         
        plt.grid(True)
        plt.axis('equal')
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


    def plot_fem_with_realworld_nodes(self, nodes_stm, deformed=False, line_thickness=0.1):
        """
        Plots the FEM results (elements and boundary conditions) along with the real-world nodes on a single plot.
    
        Parameters:
        - nodes_stm: List of Node objects (support, load, internal) to be plotted.
        - deformed: Whether to plot deformed or undeformed FEM results.
        - line_thickness: The thickness of the element boundary lines.
        """
    
        fig, ax = plt.subplots()
    
        # Setup the colormap for FEM elements (volume fractions)
        cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
        norm = Normalize(vmin=0, vmax=1)  # Normalize values from 0 to 1
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    
        # Plot FEM elements
        for n, e in enumerate(self.elements):
            if not deformed:
                coords = [node.coords for node in e.nodes]
            else:
                coords = [node.current_coords() for node in e.nodes]
    
            # Close the element by appending the first point at the end
            coords.append(coords[0])
            xs, ys = zip(*coords)
    
            # Get the color based on volume fraction (e.g., material density)
            color = scalar_map.to_rgba(self.x[n])
    
            # Fill the element with color and outline the boundary in black
            ax.fill(xs, ys, color=color, zorder=5)  # Fill element
            ax.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary
    
        # Plot boundary conditions (supports, loads, and internal nodes) from nodes_stm
        for node in nodes_stm:
            # Check if the node is a support
            if any(node.fixed):
                # Red for supports
                coords = node.current_coords() if deformed else node.coords
                ax.scatter(coords[0], coords[1], color="red", marker='x', s=100, label="Support", zorder=10)
    
            # Check if the node has non-zero forces (load)
            elif np.any(node.forces != 0):
                # Green for loads
                coords = node.current_coords() if deformed else node.coords
                ax.scatter(coords[0], coords[1], color="green", marker='x', s=100, label="Load", zorder=10)
    
            # Otherwise, it's an internal node
            else:
                # Blue for internal nodes
                coords = node.current_coords() if deformed else node.coords
                ax.scatter(coords[0], coords[1], color="blue", marker='x', s=100, label="Internal", zorder=10)
    
        # Set equal aspect ratio and enable grid for better visualization
        ax.set_aspect('equal')
        ax.grid(True)
    
        # Add a legend to distinguish supports, loads, and internal nodes
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    
        # Show the plot
        plt.show()
    
    
    
    
    
    def plot_deformed_and_undeformed(self, scale=1.0):
        """
        Plot the undeformed and deformed shape of the system elements.
    
        Parameters:
        - scale: Factor to scale the deformations for visualization purposes.
        """
        plt.figure(figsize=(10, 6))
        for element in self.elements:
            # Undeformed coordinates
            node1_coords = element.nodes[0].coords
            node2_coords = element.nodes[1].coords
            xs_undeformed = [node1_coords[0], node2_coords[0]]
            ys_undeformed = [node1_coords[1], node2_coords[1]]
    
            # Deformed coordinates using cubic shape functions for beam elements
            xi_values = np.linspace(-1, 1, 100)  # Parametric coordinates for interpolation
            xs_deformed = []
            ys_deformed = []
            for xi in xi_values:
                # Cubic shape functions for beam elements
                N1 = 1/4 * (1 - xi)**2 * (2 + xi)
                N2 = 1/4 * (1 - xi)**2 * (1 + xi)
                N3 = 1/4 * (1 + xi)**2 * (2 - xi)
                N4 = 1/4 * (1 + xi)**2 * (1 - xi)
    
                # Displacements
                node1_disp = element.nodes[0].displacements
                node2_disp = element.nodes[1].displacements
    
                # Calculate deformed coordinates using shape functions
                x_deformed = (N1 * node1_coords[0] +
                              N2 * node1_coords[0] + scale * node1_disp[0] +
                              N3 * node2_coords[0] +
                              N4 * node2_coords[0] + scale * node2_disp[0])
                y_deformed = (N1 * node1_coords[1] +
                              N2 * node1_coords[1] + scale * node1_disp[1] +
                              N3 * node2_coords[1] +
                              N4 * node2_coords[1] + scale * node2_disp[1])
    
                xs_deformed.append(x_deformed)
                ys_deformed.append(y_deformed)
    
            # Plot undeformed and deformed shapes
            plt.plot(xs_undeformed, ys_undeformed, 'b--', label='Undeformed' if element == self.elements[0] else "")
            plt.plot(xs_deformed, ys_deformed, 'r-', label='Deformed' if element == self.elements[0] else "")
    
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('System Deformation')
        plt.grid(True)
        plt.show()
        
    
    
    def plot_deformation_stm(self, scale=1.0):
        """
        Plot the undeformed and deformed shape of the system elements.

        Parameters:
        - scale: Factor to scale the deformations for visualization purposes.
        """
        plt.figure(figsize=(10, 6))
        for element in self.elements:
            # Undeformed coordinates
            node1_coords = element.nodes[0].coords
            node2_coords = element.nodes[1].coords
            xs_undeformed = [node1_coords[0], node2_coords[0]]
            ys_undeformed = [node1_coords[1], node2_coords[1]]

            # Deformed coordinates
            node1_disp = element.nodes[0].displacements
            node2_disp = element.nodes[1].displacements
            xs_deformed = [
                node1_coords[0] + scale * node1_disp[0],
                node2_coords[0] + scale * node2_disp[0]
            ]
            ys_deformed = [
                node1_coords[1] + scale * node1_disp[1],
                node2_coords[1] + scale * node2_disp[1]
            ]

            # Plot undeformed and deformed shapes
            plt.plot(xs_undeformed, ys_undeformed, 'b--', label='Undeformed' if element == self.elements[0] else "")
            plt.plot(xs_deformed, ys_deformed, 'r-', label='Deformed' if element == self.elements[0] else "")

        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('System Deformation')
        plt.grid(True)
        plt.show()

    def calculate_internal_forces(self):
        """
        Calculate the internal forces for all elements in the system.

        Returns:
        - internal_forces: A list of tuples representing the normal, shear, and moment forces for each element.
        """
        internal_forces = []
        for element in self.elements:
            k_global = element.k_e()
            internal_force = k_global @ element.displacements
            normal_force = internal_force[0]  # Axial force
            shear_force = internal_force[1]  # Shear force
            moment = internal_force[2]       # Moment
            internal_forces.append((normal_force, shear_force, moment))
        return internal_forces

    def plot_internal_forces_stm(self):
        """
        Plot the internal forces (normal, shear, and moment) for all elements in the system.
        """
        internal_forces = self.calculate_internal_forces()

        # Initialize subplots for normal, shear, and moment forces
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle('Internal Forces in Elements')

        for idx, element in enumerate(self.elements):
            node1_coords = element.nodes[0].coords
            node2_coords = element.nodes[1].coords
            center_x = (node1_coords[0] + node2_coords[0]) / 2
            center_y = (node1_coords[1] + node2_coords[1]) / 2

            # Extract internal forces
            normal_force, shear_force, moment = internal_forces[idx]

            # Plot normal forces
            axs[0].plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]], 'k-', linewidth=1)
            axs[0].text(center_x, center_y, f'N: {normal_force:.2f}', color='blue', fontsize=12, ha='center')

            # Plot shear forces
            axs[1].plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]], 'k-', linewidth=1)
            axs[1].text(center_x, center_y, f'V: {shear_force:.2f}', color='green', fontsize=12, ha='center')

            # Plot moments
            axs[2].plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]], 'k-', linewidth=1)
            axs[2].text(center_x, center_y, f'M: {moment:.2f}', color='red', fontsize=12, ha='center')

        # Set labels and titles for subplots
        axs[0].set_title('Normal Forces')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].grid(True)

        axs[1].set_title('Shear Forces')
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
        axs[1].grid(True)

        axs[2].set_title('Moments')
        axs[2].set_xlabel('X Coordinate')
        axs[2].set_ylabel('Y Coordinate')
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
