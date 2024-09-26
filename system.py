import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class System:
    def __init__(self,nodes, elements, x, penalty, x_min=1e-3):
        self.nodes = nodes
        self.elements = elements
        self.penalty = penalty
        self.x = x
        self.x_min = x_min
        self.nr_dofs = len(nodes)*2 ## assumes a continous node numbering !!

        for e in self.elements:
            e.system_penalty = penalty

        self.apply_dirichlet_bc()


    def apply_dirichlet_bc(self):
        self.fixed_dofs = []
        for n in self.nodes:
            for i,fixed in enumerate(n.fixed):
                if fixed: self.fixed_dofs.append(n.dofs[i])


    def K_global(self):
        
        K_g = np.zeros((self.nr_dofs,self.nr_dofs))
        
        n=0
        for e in self.elements:
            
            if self.x[n] < self.x_min:
                x_p = np.power(self.x_min, self.penalty) 
            else:
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

        print("---> creating Tensors")

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
    
        # Convert K_g to a sparse matrix format (Compressed Sparse Row format)
        K_g_sparse = csr_matrix(K_g)
    
        print("---> solving FE using sparse matrix solver")
        U = spsolve(K_g_sparse, F_g)
    
        # Assign the computed displacements to elements and nodes
        for e in self.elements:
            for i, dofi in enumerate(e.dofs):
                e.displacements[i] = U[dofi]
    
        for n in self.nodes:
            for i, dofi in enumerate(n.dofs):
                n.displacements[i] = U[dofi]
    
        return U


    def solve_FE_old(self):
    
        K_g, F_g = self.return_K_F_dirichlet_bc()

        print("---> solving FE")
        U =  np.linalg.solve(K_g,F_g)

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


    # def plot_fem_with_realworld_nodes(self, nodes_stm, deformed=False, line_thickness=0.1):
    #     """
    #     Plots the FEM results (elements and boundary conditions) along with the real-world nodes on a single plot.
    
    #     Parameters:
    #     - fem_data: The FEM data structure containing elements, nodes, boundary conditions, etc.
    #     - real_world_node_coordinates: List of real-world node coordinates to plot.
    #     - deformed: Whether to plot deformed or undeformed FEM results.
    #     - line_thickness: The thickness of the element boundary lines.
    #     """
    
    #     fig, ax = plt.subplots()
    
    #     # Setup the colormap for FEM elements (volume fractions)
    #     cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
    #     norm = Normalize(vmin=0, vmax=1)  # Normalize values from 0 to 1
    #     scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    
    #     # Plot FEM elements
    #     for n, e in enumerate(self.elements):
    #         if not deformed:
    #             coords = [node.coords for node in e.nodes]
    #         else:
    #             coords = [node.current_coords() for node in e.nodes]
            
    #         # Close the element by appending the first point at the end
    #         coords.append(coords[0])
    #         xs, ys = zip(*coords)
    
    #         # Get the color based on volume fraction (e.g., material density)
    #         color = scalar_map.to_rgba(self.x[n])
    
    #         # Fill the element with color and outline the boundary in black
    #         ax.fill(xs, ys, color=color, zorder=5)  # Fill element
    #         ax.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary
        
    #     if 2<0:
    #         # Plot boundary conditions (e.g., supports in red, loads in green)
    #         for node in self.nodes:
    #             if node.fixed[0] or node.fixed[1]:
    #                 # Red for supports
    #                 if not deformed:
    #                     ax.scatter([node.coords[0]], [node.coords[1]], color="red", zorder=10, label="Support" if node == self.nodes[0] else "")
    #                 else:
    #                     ax.scatter([node.current_coords()[0]], [node.current_coords()[1]], color="red", zorder=10)
    #             if abs(node.forces[0]) > 0 or abs(node.forces[1]) > 0:
    #                 # Green for loads
    #                 if not deformed:
    #                     ax.scatter([node.coords[0]], [node.coords[1]], color="green", zorder=10, label="Load" if node == self.nodes[0] else "")
    #                 else:
    #                     ax.scatter([node.current_coords()[0]], [node.current_coords()[1]], color="green", zorder=10)
    
    #     # Plot the real-world node coordinates (blue 'x' markers)
    #     real_world_xs, real_world_ys = zip(*real_world_node_coordinates)  # Unpack the coordinates into x and y lists
    #     ax.scatter(real_world_xs, real_world_ys, color="blue", marker='x', s=100, label="Real-world Nodes", zorder=15)
    
    #     # Set equal aspect ratio and enable grid for better visualization
    #     ax.set_aspect('equal')
    #     ax.grid(True)
    
    #     # Add a legend to distinguish supports, loads, and real-world nodes
    #     ax.legend(loc='best')
    
    #     # Show the plot
    #     plt.show()
    
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
    
