import numpy as np
from image_processing_utils import transformation_image_to_realworld, transformation_realworld_to_image
from extraction_utils import process_supports_and_loads
from extraction_utils import cluster_nodes, plot_all_nodes, plot_cluster_centers, nodes_on_line_support
from node import Node
from system import System
from beam_element import BeamElement
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class STM:
    def __init__(self, s, image, dimensions, dimensions_img):
        self.node_list = []
        self.element_list = []
        self.adjacency_list = []
        self.image = image
        self.system_to = s
        self.system_stm = None
        self.dimensions = dimensions
        self.dimensions_img = dimensions_img
        self.node_list_bc = None


    def generate_stm_system(self):
        # pay respect to new rotational DOF per node 
        for node in self.node_list:
            node.forces = np.append(node.forces, 0)
            fixed = [node.fixed[0], node.fixed[0], False]
            node.fixed = fixed
            node.displacements = np.zeros(3)
            
            
        # with the adjacency matrix and node coords generate beam_list
        if len(self.element_list) == 0:
            for conn in self.adjacency_list:
                nodes= [self.node_list[conn[0]], self.node_list[conn[1]]]
                self.element_list.append(BeamElement(nodes))
                
        x = np.ones((len(self.element_list))) 
        
        
        system_stm = System(self.node_list, self.element_list, x)
        
        # apply dirichlet BCs
        system_stm.apply_dirichlet_bc()
        
        
        self.system_stm = system_stm
        
        return system_stm


    def extract_bcs(self):
        # extract BCs from system
        line_supports, line_loads, nodes_stm_bc = process_supports_and_loads(self.system_to)
        
        # add point supports and loads to stm
        i = len(self.node_list)
        for node in nodes_stm_bc:
            node.id = i
            node.dofs = [3*i, 3*i+1, 3*i+2]
            node.coords_img = transformation_realworld_to_image(node.coords, self.dimensions, self.dimensions_img)
            self.node_list.append(node)

        # transform support lines to image space
        support_lines_img = []
        for line in line_supports:
            line_img = [transformation_realworld_to_image(point, self.dimensions, self.dimensions_img) for point in line]
            support_lines_img.append(line_img)
        
        # transform load lines to image space
        load_lines_img = []
        for line in line_loads:
            line_img = [transformation_realworld_to_image(point, self.dimensions, self.dimensions_img) for point in line]
            load_lines_img.append(line_img)

        
        # find points on lines and add them to the stm
        if len(support_lines_img)>0:
            # find pixels on line supports with neighboring black pixels
            node_candidates = nodes_on_line_support(self.image, support_lines_img)
            
            # Perform clustering and get the cluster centers
            eps = 5  # Maximum distance for points to be considered in the same cluster
            min_samples = 2  # Minimum number of points required to form a cluster
            cluster_centers, labels = cluster_nodes(node_candidates, eps=eps, min_samples=min_samples)
       
            i = len(self.node_list)
            for coords_img in cluster_centers:
                coords = transformation_image_to_realworld(coords_img, self.dimensions, self.dimensions_img)
                node = Node(coords, i, [3*i, 3*i+1, 3*i+2], fixed=[True, True])
                node.coords_img = coords_img
                self.node_list.append(node)
                i += 1
               
        self.node_list_bc = self.node_list.copy() # Backup
        
        #plot all nodes extracted boundary nodes
        plt.imshow(self.image, cmap='gray')

        for node in self.node_list:
            if np.any(node.forces != 0):
                plt.scatter(node.coords_img[0], node.coords_img[1], color='green', label='loads', marker='x', s=100)
            elif any(node.fixed):
                plt.scatter(node.coords_img[0], node.coords_img[1], color='red', label='supports', marker='x', s=100)
             

        # Ensure only unique entries in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Removes duplicate labels by using a dictionary

        # Set plot settings
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.title("Extracted Nodes from BCs")
        plt.show()

        
    def nodes_skel(self, skel_image, node_candidates):
        
        # ensure only boundary nodes are stored at this point
        self.node_list = self.node_list_bc.copy() 
        
        
        # map boundary nodes to skeleton
        for node in self.node_list:
            cx, cy = map(int, node.coords_img)  # Node coordinates as integers
            
            detected = False
            distance = 1  # Start with a small radius
            
            while not detected:
                # Circular search: iterate over a grid of points within a square, but only keep points within the circle radius
                for dx in range(-distance, distance + 1):
                    for dy in range(-distance, distance + 1):
                        # Check if the point is within the current radius (circular search)
                        if np.sqrt(dx**2 + dy**2) <= distance:
                            nx, ny = cx + dx, cy + dy  # Neighbor coordinates

                            # Ensure the coordinates are within the image bounds
                            if 0 <= nx < skel_image.shape[1] and 0 <= ny < skel_image.shape[0]:
                                # Check if the neighbor is a black pixel (skeleton pixel = 1)
                                if skel_image[ny, nx] == 1:
                                    node.coords_skel = [nx, ny]
                                    detected = True
                                    break

                    if detected:
                        break
                
                distance += 1  # Increment the search radius if no pixel was found in the current range

        
        # add the extracted nodes to the stm
        i = len(self.node_list)
        for coords_img in node_candidates:
            coords = transformation_image_to_realworld(coords_img, self.dimensions, self.dimensions_img)
            node = Node(coords, i, [3*i, 3*i+1, 3*i+2])
            node.coords_img = coords_img
            node.coords_skel = coords_img
            self.node_list.append(node)
            i += 1
               
  
    def plot_truss_structure(self, skeletonized_image):
        """
        Visualizes the truss structure by drawing straight lines between connected nodes in a similar style to the provided reference code.
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # Plot the first image with nodes in image space
        ax[0].imshow(self.image)
        ax[0].set_title("Strut and Tie Model on TopOpt Results")

        # Plot the nodes as blue "x" markers with labels
        for node in self.node_list:
            coords = tuple(map(int, node.coords_img))
            ax[0].scatter(coords[0], coords[1], color='blue', marker='x', s=100, label="Nodes" if node.id == 0 else None)
            ax[0].text(coords[0], coords[1], str(node.id), color='blue', fontsize=14)

        # Plot the trusses as red lines
        for conn in self.adjacency_list:
            start, end = conn[0], conn[1]
            start = self.node_list[start].coords_img
            end = self.node_list[int(end)].coords_img
            ax[0].plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2, label="Trusses")

        # Remove duplicate labels in the legend for the first plot
        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

        # Plot the second image with nodes in skeleton space
        ax[1].imshow(skeletonized_image, cmap='gray')
        ax[1].set_title("Strut and Tie Model on Skeleton Nodes")

        # Plot the nodes as blue "x" markers with labels in skeleton space
        for node in self.node_list:
            coords_skel = tuple(map(int, node.coords_skel))
            ax[1].scatter(coords_skel[0], coords_skel[1], color='blue', marker='x', s=100, label="Nodes" if node.id == 0 else None)
            ax[1].text(coords_skel[0], coords_skel[1], str(node.id), color='blue', fontsize=14)

        # Plot the trusses as red lines in skeleton space
        for conn in self.adjacency_list:
            start, end = conn[0], conn[1]
            start = self.node_list[start].coords_skel
            end = self.node_list[int(end)].coords_skel
            ax[1].plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2, label="Trusses")

        # Set equal aspect ratio, grid, and display for both subplots
        for a in ax:
            a.set_aspect('equal', adjustable='box')
            a.grid(True)
            a.axis('off')

        plt.tight_layout()
        plt.show()
        
        
    def plot_fem_with_realworld_nodes(self, deformed=False, line_thickness=0.1):
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
        for n, e in enumerate(self.system_to.elements):
            if not deformed:
                coords = [node.coords for node in e.nodes]
            else:
                coords = [node.current_coords() for node in e.nodes]
    
            # Close the element by appending the first point at the end
            coords.append(coords[0])
            xs, ys = zip(*coords)
    
            # Get the color based on volume fraction (e.g., material density)
            color = scalar_map.to_rgba(self.system_to.x[n])
    
            # Fill the element with color and outline the boundary in black
            ax.fill(xs, ys, color=color, zorder=5)  # Fill element
            ax.plot(xs, ys, color="black", zorder=6, linewidth=line_thickness)  # Element boundary
    
        # Plot boundary conditions (supports, loads, and internal nodes) from nodes_stm
        for node in self.node_list:
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
    
        # Plot trusses
        for conn in self.adjacency_list:
            start, end = conn[0], conn[1]
            start = self.node_list[start].coords
            end = self.node_list[int(end)].coords
            ax.plot([start[0], end[0]], [start[1], end[1]], color='yellow', linewidth=1, zorder=20, label="Trusses")
    
        # Set equal aspect ratio and enable grid for better visualization
        ax.set_aspect('equal')
        ax.grid(True)
    
        # Add a legend to distinguish supports, loads, and internal nodes
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
        # Show the plot
        plt.show()
