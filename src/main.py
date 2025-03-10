import json
import sys
import os
from tkinter import Tk
import tkinter as tk
from tkinter import messagebox
import calfem.geometry as cfg
import calfem.mesh as cfm
import numpy as np

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parameterGUI import ParameterInputGUI 
from src.geometryGUI import GeometryInputGUI
from utils.config import load_config
from system import System


from mesh import Mesh
from system import System
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
import numpy as np
from mesh import Mesh


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


class GeometryInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Geometry Input")

        self.canvas = tk.Canvas(master, width=400, height=400, bg="white")
        self.canvas.pack()

        self.points = []
        self.lines = []
        self.surfaces = []

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.create_surface)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

        self.node_list = None
        self.element_list = None

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")

        if len(self.points) > 1:
            self.lines.append((self.points[-2], self.points[-1]))
            self.canvas.create_line(self.points[-2], self.points[-1])

    def create_surface(self, event):
        if len(self.points) < 3:
            messagebox.showerror("Input Error", "At least 3 points are required to create a surface.")
            return

        # Automatically close the surface by connecting the last point to the first point
        self.lines.append((self.points[-1], self.points[0]))
        self.canvas.create_line(self.points[-1], self.points[0])

        self.surfaces.append(self.points)
        self.points = []

    def submit(self):
        if not self.surfaces:
            messagebox.showerror("Input Error", "No surfaces created.")
            return

        g = cfg.Geometry()

        for surface in self.surfaces:
            print(f"Creating surface with points: {surface}")  # Debug print
            for i, (x, y) in enumerate(surface):
                if x is None or y is None:
                    print(f"Skipping invalid point: ({x}, {y})")  # Debug print
                    continue
                print(f"Adding point: ({x}, {y}), ID={i}")  # Debug print
                g.point([x, y], ID=i)
                num_points =i
                
            if num_points < 2:
                print(f"Skipping surface creation due to insufficient points: {num_points}")  # Debug print
                continue

            for i in range(num_points):
                print(f"Adding spline: ({i}, {(i + 1) }), ID={i}")  # Debug print
                try:
                    g.spline([i, (i + 1) ], ID=i)
                except Exception as e:
                    print(f"Exception occurred while adding spline ({i}, {(i + 1) }): {e}")  # Debug print
                    continue

            # close the surface
            try:
                print(f"Adding spline: ({num_points}, 0), ID={num_points}")  # Debug print
                g.spline([num_points, 0], ID=num_points)
            except Exception as e:
                print(f"Exception occurred while adding spline ({num_points}, 0): {e}")  # Debug print
                continue

            #print(f"Creating surface with points: {len(self.points)}")  # Debug print
            try:
                print(f"Creating surface with lines: {list(range(num_points+1))}")  # Debug print
                g.surface((list(range(num_points+1))))
            except Exception as e:
                #print(f"Exception occurred while creating surface with points {point_ids}: {e}")  # Debug print
                continue

        cfv.drawGeometry(g)
        cfv.showAndWait()
        
        mesh = cfm.GmshMesh(g)
        mesh.elType = 3
        mesh.dofsPerNode = 2
        mesh.elSizeFactor = 10

        try:
            print("Creating mesh...")  # Debug print
            coords, edof, dofs, bdofs, elementmarkers = mesh.create()
            self.node_list, self.element_list = Mesh.create(coords, dofs, edof)
            print('number of elements:', len(self.element_list)) 
        except Exception as e:
            print(f"Exception occurred while creating mesh: {e}")  # Debug print
            return

        # Save the mesh data or pass it to the next step
        # For example, save to a file or pass to another function

        self.master.destroy()


def main():
    
    # Initialize the GUI for parameter input
    root = Tk()
    param_gui = ParameterInputGUI(root)
    root.mainloop()

    # Initialize the GUI for geometry input
    root = tk.Tk()
    geom_gui = GeometryInputGUI(root)
    root.mainloop()

    # Access node_list and element_list from the GeometryInputGUI instance
    node_list = geom_gui.node_list
    element_list = geom_gui.element_list

    if node_list is None or element_list is None:
        print("Error: node_list or element_list is None")
        return
    

    # # Load the geometry data
    # g = cfg.Geometry()

    # g.point([0.0, -1.0], ID=0) # point 0
    # g.point([4.0, -1.0], ID=1) # point 1
    # g.point([4.0, 1.0], ID=2) # point 2
    # g.point([0.0, 1.0], ID=3) # point 3

    # g.spline([0, 1], ID=0) # line 0
    # g.spline([1, 2], ID=1) # line 1
    # g.spline([2, 3], ID=2) # line 2
    # g.spline([3, 0], ID=3) # line 3

    # g.surface([0, 1, 2, 3])

    # mesh = cfm.GmshMesh(g)

    # # Set the mesh parameters
    # mesh.elType = 3 
    # mesh.dofsPerNode = 2     
    # mesh.elSizeFactor = 0.1

    # coords, edof, dofs, bdofs, elementmarkers = mesh.create()
    # node_list, element_list = Mesh.create(coords, dofs, edof)

    #  The Parameter GUI saves the parameters to a file, load them
    parameters = load_config('config/parameters.json')

    # Set the material properties
    for e in element_list:
        e.E = parameters['Youngs_modulus']
        e.nu = parameters['Poissons_ratio']

    # Create the system with the loaded parameters and geometry data
    system = System(node_list, element_list, parameters)

    # BC
    system.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    system.load_point([4,-1],[0,-10])
    
    system.apply_dirichlet_bc()

    system.plot2(deformed=False)

    # Run the FEA
    system.solve_FE_sparse()
    # Visualize the results
    system.plot2(deformed=True)


    # # Or run the optimization
    # dv = np.ones(len(system.elements))
    # H_f = convolution_operator(system)
    # system_optimized = top_opt(system, H_f, dv, parameters['max_iteration'])
    # # combined plot of optimized structure, objecitve history and element density distribution
    # system_optimized.combined_plot()

if __name__ == "__main__":
    main()