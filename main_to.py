import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import copy
import cv2

from mesh import Mesh
from system import System


# # Optimized systems used in the Thesis
# from examples import create_mesh_cantilever0, create_mesh_cantilever1, create_mesh_corbel, create_mesh_wall_with_openings
# # Other systems with non optimized parameters
# from examples import  create_mesh_cantilever_short, create_mesh_cantilever1_hole



###
# From GUI
###

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
import numpy as np
from mesh import Mesh

g = cfg.Geometry()

g.point([0.0, -1.0], ID=0) # point 0
g.point([4.0, -1.0], ID=1) # point 1
g.point([4.0, 1.0], ID=2) # point 2
g.point([0.0, 1.0], ID=3) # point 3


g.spline([0, 1], ID=0) # line 0
g.spline([1, 2], ID=1) # line 1
g.spline([2, 3], ID=2) # line 2
g.spline([3, 0], ID=3) # line 3


hole = False

if hole:
    g.point([1.0, 0.5], ID=4)
    g.point([2.0, 0.5], ID=5)
    g.point([2.0, -0.5], ID=6)
    g.point([1.0, -0.5], ID=7)
    g.bspline([4,5,6,7,4], ID=4)
    g.surface([0, 1, 2, 3], [[4]])
else:
    g.surface([0, 1, 2, 3])

mesh = cfm.GmshMesh(g)

mesh.elType = 3 
mesh.dofsPerNode = 2     
mesh.elSizeFactor = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()

node_list, element_list = Mesh.create(coords, dofs, edof)

# # Nesh 
# node_list, element_list = 

# Definition of supports
# point_supports, line_supports = 

# Definition of loads
# point_loads, line_loads = 



###
# From configs file
###

# Define the path to save the results. If you dont want to save them set the path to None
path = "C:/Users/luziu/Documents/GitHub/TopOpt-deepBeams-LuziusMoll"

# Definition of the systems parameters:
name = 'cantilever0'
volfrac=0.4
penalty = 3
x_min = 1e-3 
r_min = 0.15  #0.25

# TopOpt parameters
max_iteration = 50
mesh_ind_filter = True


# Set up FE problem
s = System(node_list, element_list, r_min=r_min, volfrac=volfrac, penalty=penalty, x_min=x_min)


###
# Should happen automatically with a s.init something like this
###

# BC
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
s.load_point([4,-1],[0,-1])
s.apply_dirichlet_bc()


for e in element_list:
    e.E = 30000
    e.nu = 0.15

# volume fraction for all elements is set to volfrac
x = np.ones(len(element_list),dtype=float)*volfrac

s.name = f"{name}_N{len(element_list)}_r{r_min}_p{penalty}"

if mesh_ind_filter == False:
    s.name = f"{s.name}_no_filter"



#%% Convolution operator for mesh independency filtering
""" from sigmund2001: A 99 line topology optimization code written in Matlab: eq6"""


def convolution_operator(s):
    # distance between current element and all others
    element_centers = s.element_centers()
    element_centers = np.array(element_centers)
    
    dist = []
    for i in range(len(s.elements)):
        dist_ij = []
        for j in range(len(s.elements)):
            dist_x = element_centers[i,0]-element_centers[j,0]
            dist_y = element_centers[i,1]-element_centers[j,1]
            dist_ij.append(np.sqrt(dist_x**2 + dist_y**2))
        dist.append(dist_ij)
    
        
    # convolution operator H_f
    H_f = r_min * np.ones([len(s.x),len(s.x)]) - dist
    # set negativ values (elements outside of r_min) to zero
    H_f[H_f < 0] = 0
    
    return H_f


#%% Optimality criteria
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(x,volfrac,dc,dv):
    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(len(x))
    while (l2-l1)/(l1+l2)>1e-8:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # possibility to define passive areas in regular mesh
        if 2<0: # for regular mesh only 
            for ely in range(40):
                for elx in range(80):
                    if np.sqrt((ely-20)**2 + (elx-30)**2) < 10:
                        xnew[elx*40+ely] = x_min
        
        # if np.mean(dv*xnew)> np.mean(dv*volfrac):
        if np.mean(xnew)> volfrac:   # this assumes that all elements have a comparable area. If that is not the case, a scaling with the element areas is necessary
            l1=lmid
        else:
            l2=lmid
            
        # with out this float division by 0 can occour in the while loop criteria (additional line compared to sigmund 200 line implementation)
        if l1 + l2 == 0:
            return xnew
        
    
    return xnew


#%% Actual optimization 
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """


def top_opt(s, H_f, dv, max_iteration):
    # Set loop counter and gradient vectors 
    loop=0
    obj_hist = []
    change=1

    # The following must be initialized to use the NGuyen/Paulino OC approach
    xold=s.x.copy()
    obj_change = 1

    
    
    while obj_change > 0.000001 and loop < max_iteration:  # my own criteria
        loop = loop + 1
    
        # Solve FE problem
        u = s.solve_FE_sparse()
        
        # Objective and sensitivity
        obj = s.compliance()
        obj_hist.append(obj)
        if len(obj_hist) > 1:
            obj_change = abs(obj_hist[loop - 1] - obj_hist[loop - 2]) / obj_hist[loop - 1]
        # according to sigmund2001 eq4 (no filter)
        dc = s.sensitivity_compliance()
        
        # according to sigmund2001 eq5 (with filter)
        if mesh_ind_filter:
            dc_filtered = []
            for i in range(len(s.elements)):
                # additional if criteria compared to sigmund
                if x[i] * np.sum(H_f[:, i]) > 0:
                    dc_filtered_i = 1 / x[i] * np.sum(H_f[:, i]) * np.sum(H_f[:, i] * x * dc)
                else:
                    dc_filtered_i = dc[i]
                dc_filtered.append(dc_filtered_i)
    
            dc = dc_filtered
    
        # Optimality criteria
        xold[:] = x
        x[:] = oc(x, volfrac, dc, dv)
    
        # pass new x vector to system
        s.x = x
    
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(len(s.elements), 1) - xold.reshape(len(s.elements), 1), np.inf)
    
        if (loop) % 5 == 0 or loop==1:
            print('Iteration:', loop)
            print('obj:',obj)
            print('mean x:',np.mean(x))
            s.plot2(deformed=False, disp_bc=False, line_thickness=0.2)    
    
    s.obj_hist = obj_hist
    
    # combined plot of optimized structure, objecitve history and element density distribution
    s.combined_plot()


# Run the optimization
dv = np.ones(len(s.elements))
H_f = convolution_operator(s)
top_opt(s, H_f, dv, max_iteration)



# save as pickle
if path is not None:
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # Define the file name
    full_path = os.path.join(path, f"{s.name}.pkl")
    
    # Save the object
    with open(full_path, "wb") as file:
        pickle.dump(s, file)
    
    print(f"System saved successfully to {full_path}")
    



