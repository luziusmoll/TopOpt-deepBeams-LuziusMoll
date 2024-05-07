import numpy as np
from membrane import QuadPlateMembrane
import matplotlib.pyplot as plt
from mesh_test import create_mesh as create_mesh 
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns




class node:
    def __init__(self, coords, id, dofs, fixed = [False,False], forces = np.zeros(2)) -> None:
        self.coords = coords
        self.id = id
        self.dofs = dofs
        self.forces = forces
        self.fixed = fixed
        self.displacements = np.zeros(2)

    def current_coords(self):
        return self.coords + self.displacements
        
class element:
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
        
        ## for regular/easy mesh 
        #E = 1.0
        #nu = 0.3
        #k = np.array([
        #    1.0/2.0-nu/6.0, 1.0/8.0+nu/8.0, -1.0/4.0-nu/12.0, -1.0/8.0+3.0*nu/8.0,
        #    -1.0/4.0+nu/12.0, -1.0/8.0-nu/8.0, nu/6.0, 1.0/8.0-3.0*nu/8.0
        #])

        #k_e = E / (1.0-np.power(nu,2.0)) * np.array([
        #    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        #    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        #    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        #    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        #    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        #    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        #    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        #    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        #    ])
        
        q_e = QuadPlateMembrane(self.nodes)
        k_e = q_e.calculate_elastic_stiffness_matrix()
        
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
    
    def compliance_try(self,x):
        u = self.displacements
        f = self.k_e()@self.displacements
        g = (u[0] + u[1] - u[2] - u[3]) *x**(self.system_penalty) * np.sum(f[:4]) + (u[4] + u[5] - u[6] - u[7]) * x**(self.system_penalty) * np.sum(f[4:8])
        return g

    def sensitivity_compliance(self,x):
        """
        from sigmund2001
        A 99 line topology optimization code written in Matlab
        eq4 
        """
        dc_e = self.k_e()@self.displacements
        dc_e = self.displacements@dc_e
        #self.dc = dc_e * (-self.system_penalty) * np.power(x,self.system_penalty-1.0)
        return dc_e * (-self.system_penalty) * np.power(x,self.system_penalty-1.0) 
    

    def sensitivity_compliance_try(self,x):
        u = self.displacements
        f = self.k_e()@self.displacements
        dg = (u[0] + u[1] - u[2] - u[3]) * self.system_penalty * x**(self.system_penalty-1) * np.sum(f[:4]) + (u[4] + u[5] - u[6] - u[7]) * self.system_penalty * x**(self.system_penalty-1) * np.sum(f[4:8])
        return -dg


class system:
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
        
            E = np.power(x[n], self.penalty)
            if E < self.E_min:
                E=self.E_min 
            k = e.k_e() * E

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
           sum_c +=  e.compliance(x[n]) 
           n+=1
        return sum_c
        #return sum([e.compliance(x) for e in self.elements])
    
    def sensitivity_compliance(self):
        """
        from sigmund2001
        A 99 line topology optimization code written in Matlab
        eq4
        """
        dc=[]
        n=0
        for e in self.elements:
            dc.append(e.sensitivity_compliance(x[n]))
            n+=1
        return dc


    # def sensitivity_compliance(self):
    #     for e in self.elements:
    #         e.sensitivity_compliance()    

    # def find_and_return_nearest_node(self,search_coords):
    #     min_dist=10e10
    #     nearest_node = self.nodes[0]

    #     for i, n_i in enumerate(self.nodes):

    #         distance = np.linalg.norm(search_coords-n_i.coords)
    #         if distance<min_dist:
    #             min_dist=distance
    #             nearest_node = n_i

    #     return nearest_node
    
    # def fix_node_by_coord(self,fix_coord,fix=[True,True]):
    #     self.find_and_return_nearest_node(fix_coord).fixed = fix


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
        
        # Iterate over all nodes to find the node at the specified coordinates
        for n in self.nodes:
            # Calculate the distance from the current node's coordinates to the load coordinates
            if np.linalg.norm(n.coords - load_coord) <= tol:
                n.forces = force
                #print(f"Load applied to node at {n.coords} with force {force}")
                return True

        print("No node found within tolerance to apply the load.")
        return False


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
            color = scalar_map.to_rgba(x[n])

            # Fill element with appropriate color and outline in black
            plt.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac
            plt.plot(xs, ys, color="black", zorder=6)  # Element boundary in black
            n+=1

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
         
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
class mesh:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create():   

        [ex, ey], coords, dofs, edof = create_mesh()
        nr_nodes = len(dofs)
        nr_elements = len(edof)

        node_list = []
        for i in range(nr_nodes):
            node_dofs = [d-1 for d in dofs[i]]
            node_list.append(node(coords[i],id=i,dofs=node_dofs))


        element_list = []
        for i in range(nr_elements):
            mesh_element_dofs = [d-1 for d in edof[i]]
            
            node_element_list = []
            for n in node_list:
                if n.dofs[0] == mesh_element_dofs[0]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[2]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[4]:
                    node_element_list.append(n)
                elif n.dofs[0] == mesh_element_dofs[6]:
                    node_element_list.append(n)

                if len(node_element_list) == 4:
                    break

            sorted_node_element_list = []
            
            for i in range(8):
                if i%2!=0: continue
                for n in node_element_list:
                    if n.dofs[0]==mesh_element_dofs[i]:
                        sorted_node_element_list.append(n)
                        break
                
            element_list.append(element(sorted_node_element_list)) 

        return node_list, element_list 


#%% defining the probelm and solving FE for initial configuration

# geometrie is defined in mesh_test and called by mesh.create()
node_list, element_list  = mesh.create()
print('number of elements:', len(element_list))

# volume fraction for all elements is set to 1
x = np.ones(len(element_list),dtype=float)

# setting up the system
s = system(node_list, element_list, x, penalty=3)

#s.find_and_return_nearest_node(np.array([6.0,3.0])).forces = np.array([-40.0,0.0])/10e2
#s.fix_node_by_coord(np.array([0.0,0.0]),[True,True])

s.fix_line(np.array([0.0,0.0]), np.array([0.0,1.0]))
#s.load_line(np.array([4.0,0.5]), np.array([4.0,-0.5]),forces=np.array([0.0,-1.0])/20e3)
s.load_point([4,0],[0.1,0])
s.apply_dirichlet_bc()


# solve for initial x vector
u = s.solve_FE()
obj = s.compliance()
dc = s.sensitivity_compliance()
s.plot(deformed=False)
s.plot(deformed=True)



#%% TopOpt from DTU code
"""
origin: DTU
name: minimum compliance problem (basic 200 lines python code)
source: https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python 
"""

# parameters:
volfrac=0.4
penalty = 3
E_min = 0
ft=0        # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
r_min = 0.3
max_iteration = 20 
mesh_ind_filter = False

# set up geometry as defined in mesh_test
node_list, element_list  = mesh.create()

# Set up FE problem
s = system(node_list, element_list, x, penalty, E_min)
s.fix_line(np.array([0.0,0.0]), np.array([0.0,1.0]))
#s.load_line(np.array([4.0,0.5]), np.array([4.0,-0.5]),forces=np.array([0.0,-1.0])/20e3)
s.load_point([4,0],[0.1,0])
s.apply_dirichlet_bc()


# calculate convolution operator for mesh independency filtering
"""
from sigmund2001
A 99 line topology optimization code written in Matlab
eq6
"""
# distance between current element and all others
element_centers = s.element_centers()
element_centers = np.array(element_centers)

dist = []
for i in range(len(element_list)):
    dist_ij = []
    for j in range(len(element_list)):
        dist_x = element_centers[i,0]-element_centers[j,0]
        dist_y = element_centers[i,1]-element_centers[j,1]
        dist_ij.append(np.sqrt(dist_x**2 + dist_y**2))
    dist.append(dist_ij)

    
# convolution operator H_f
H_f = r_min * np.ones([len(x),len(x)]) - dist
# set negativ values (elements outside of r_min) to zero
H_f[H_f < 0] = 0

        
       

x=volfrac * np.ones(len(element_list),dtype=float)
xold=x.copy()
xPhys=x.copy()
g=0 # must be initialized to use the NGuyen/Paulino OC approachgls
# Optimality criterion
def oc(n_ele,x,volfrac,dc,dv,g):
    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.1 
    # reshape to perform vector operations
    xnew=np.zeros(n_ele)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (xnew,gt)


# Set loop counter and gradient vectors 
loop=0
obj_hist = []
change=1
dv = np.ones(len(element_list))
dc = np.ones(len(element_list))
ce = np.ones(len(element_list))
while change>0.0001 and loop<max_iteration: 
    loop=loop+1
    
    # Solve FE problem
    u = s.solve_FE() 
    
    #K_g = s.K_global()
    #print(K_g)
    # Objective and sensitivity
    obj=s.compliance()
    obj_hist.append(obj)
    # according to sigmund2001 eq4 (no filter)
    dc=s.sensitivity_compliance()  
    
    # according to sigmund2001 eq5 (with filter)
    if mesh_ind_filter == True:
        dc_filtered = []
        for i in range(len(element_list)):
            dc_filtered_i = 1 / x[i] * np.sum(H_f[:,i]) * np.sum( H_f[:,i] * x * dc)
            dc_filtered.append(dc_filtered_i)
            
        dc= dc_filtered
        
    
    dv = np.ones(len(element_list))
    # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
    # if ft==0:
    #     dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
    # elif ft==1:
    #     dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
    #     dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
    # Optimality criteria
    xold[:]=x
    (x[:],g)=oc(len(element_list),x,volfrac,dc,dv,g)
    print(loop)
    # Filter design variables
    # if ft==0:   xPhys[:]=x
    # elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
    # Compute the change by the inf. norm 
    change=np.linalg.norm(x.reshape(len(element_list),1)-xold.reshape(len(element_list),1),np.inf)
    # Plot to screen
    # im.set_array(-xPhys.reshape((nelx,nely)).T)
    # fig.canvas.draw()
    # Write iteration history to screen (req. Python 2.6 or newer)
    print('obj:',obj)
    print('change:', change)
    print('mean x:',np.mean(x))
    #print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    #s.plot2(deformed=False)
    s.plot2(deformed=True)
 


# Plotting the objective history
plt.figure()
plt.plot(obj_hist)
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.title('Objective History')
plt.grid(True)
plt.show()


# Plotting the distribution of x
plt.hist(x, bins=30, alpha=0.75)
plt.title('Histogram of x')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



#%% other plots 


# Plotting stresses or comlpiances
# define what you want to plot as x
# x=[]
# for e in element_list:
#     x.append(e.forces_element(x))

# X=np.array(x)  
# max_x = np.max(x) 
#x = np.mean(abs(x), axis=1) 

 
# for i in range(8):  
#     x=X[:,i]
    
#     x=x/max_x
    
#     # Setup the colormap
#     cmap = plt.cm.gray_r  # Uses inverted grayscale where 0 is white, 1 is black
#     norm = Normalize(vmin=0, vmax=1)  # Normalize x from 0 to 1
#     scalar_map = ScalarMappable(norm=norm, cmap=cmap)
#     n = 0
#     for e in element_list:
#         coords = [n.coords for n in e.nodes]
        
#         # Ensure the element is closed by adding the first point at the end
#         coords.append(coords[0])
#         xs, ys = zip(*coords)
    
#         # Get color based on volume fraction
#         color = scalar_map.to_rgba(x[n])
    
#         # Fill element with appropriate color and outline in black
#         plt.fill(xs, ys, color=color, zorder=5)  # Fill color based on volfrac
#         plt.plot(xs, ys, color="black", zorder=6)  # Element boundary in black
#         n+=1
    
#     print("---> plotting bcs")
#     for n in node_list:
#         if n.fixed[0] or n.fixed[1]:
#             plt.scatter([n.current_coords()[0]],[n.current_coords()[1]],color="red",zorder=10)
        
#         if abs(n.forces[0])>0 or abs(n.forces[1])>0:
#             plt.scatter([n.coords[0]],[n.coords[1]],color="green",zorder=10)
        
#     plt.grid(False)
#     plt.axis('equal')
#     plt.show()
    
