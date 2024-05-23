import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec



class System:
    def __init__(self,nodes,elements,x,penalty, E_min=1e-9):
        self.nodes = nodes
        self.elements = elements
        self.penalty = penalty
        self.x = x
        self.E_min = E_min
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
        
            E = np.power(self.x[n], self.penalty)
            n+=1
            if E < self.E_min:
                E=self.E_min 
            k = np.multiply(e.k_e(), E)

            for i, dof_i in enumerate(e.dofs):
                for j, dof_j in enumerate(e.dofs):
                    K_g[dof_i,dof_j] += k[i,j] 
      
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
        """
        from sigmund2001
        A 99 line topology optimization code written in Matlab
        eq4
        """
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
        
        # # Iterate over all nodes to find the node at the specified coordinates
        # for n in self.nodes:
        #     # Calculate the distance from the current node's coordinates to the load coordinates
        #     if np.linalg.norm(n.coords - load_coord) <= tol:
        #         n.forces = force
        #         #print(f"Load applied to node at {n.coords} with force {force}")
        #         return True

        # print("No node found within tolerance to apply the load.")
        # return False
        self.find_and_return_nearest_node(load_coord).forces = force


    def plot(self, deformed=False):
        

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
            plt.plot(xs,ys,color="black",zorder=6)

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
        
    def plot2(self, deformed=False):
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
            plt.plot(xs, ys, color="black", zorder=6)  # Element boundary in black
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
    
    def plot3(self, ax, deformed=False):
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
            ax.plot(xs, ys, color="black", zorder=6)  # Element boundary in black
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