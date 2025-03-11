import json
import sys
import os
import tkinter as tk
import numpy as np


# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parameterGUI import ParameterInputGUI 
from src.geometryGUI import GeometryInputGUI
from src.utils.config import load_config
from src.system import System
from src.system_setup import SystemSetup



# Convolution operator for mesh independency filtering
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
    H_f = s.r_min * np.ones([len(s.x),len(s.x)]) - dist
    # set negativ values (elements outside of r_min) to zero
    H_f[H_f < 0] = 0
    
    return H_f


# Optimality criteria
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(x,volfrac,dc,dv,x_min):
    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(len(x))
    while (l2-l1)/(l1+l2)>1e-8:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # if np.mean(dv*xnew)> np.mean(dv*volfrac):
        if np.mean(xnew)> volfrac:   # this assumes that all elements have a comparable area. If that is not the case, a scaling with the element areas is necessary
            l1=lmid
        else:
            l2=lmid
            
        # with out this float division by 0 can occour in the while loop criteria (additional line compared to sigmund 200 line implementation)
        if l1 + l2 == 0:
            return xnew
        
    return xnew


# Actual optimization 
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def top_opt(s, H_f, dv, max_iteration):
    # Set loop counter and gradient vectors 
    loop=0
    obj_hist = []
    change=1

    # The following must be initialized to use the NGuyen/Paulino OC approach
    x = s.x.copy()
    xold = s.x.copy()
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
        x[:] = oc(x, s.volfrac, dc, dv,s.x_min)
    
        # pass new x vector to system
        s.x = x
    
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(len(s.elements), 1) - xold.reshape(len(s.elements), 1), np.inf)
    
        if (loop) % 5 == 0 or loop==1:
            print('Iteration:', loop)
            print('obj:',obj)
            print('mean x:',np.mean(x))
            #s.plot2(deformed=False, disp_bc=False, line_thickness=0.2)    
    
    s.obj_hist = obj_hist

    return s


def main():
    
    # Initialize the GUI for geometry input
    root = tk.Tk()
    geom_gui = GeometryInputGUI(root)
    root.mainloop()

    # Initialize the GUI for parameter input
    root = tk.Tk()
    param_gui = ParameterInputGUI(root)
    root.mainloop()

    # Get node_list and element_list system setup
    system_setup = SystemSetup()
    node_list, element_list = system_setup.create_mesh_from_geometry()

    if node_list is None or element_list is None:
        print("Error: node_list or element_list is None")
        return
    
    #  The Parameter GUI saves the parameters to a file, load them
    parameters = load_config('config/parameters.json')

    # Set the material properties
    for e in element_list:
        e.E = parameters['Youngs_modulus']
        e.nu = parameters['Poissons_ratio']
        

    print('number of elements:', len(element_list))

    # Create the system with the loaded parameters and geometry data
    system = System(node_list, element_list, parameters)

    # Apply boundary conditions
    system = system_setup.apply_boundary_conditions(system)


    # system.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    # system.load_point([4,-1],[0,-10])
    
    # system.apply_dirichlet_bc()

    # system.plot2(deformed=False)

    # # Run the FEA
    # system.solve_FE_sparse()
    # # Visualize the results
    # system.plot2(deformed=True)


    # Or run the optimization
    dv = np.ones(len(system.elements))
    H_f = convolution_operator(system)
    system_optimized = top_opt(system, H_f, dv, parameters['max_iteration'])
    # combined plot of optimized structure, objecitve history and element density distribution
    system_optimized.combined_plot()

if __name__ == "__main__":
    main()