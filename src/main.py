import json
import sys
import os
from tkinter import Tk

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gui import ParameterInputGUI #, GeometryInputGUI
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


def main():
    
    # Initialize the GUI for geometry input
    root = Tk()
    gui = ParameterInputGUI(root)
    root.mainloop()

    # Load the geometry data
    g = cfg.Geometry()

    g.point([0.0, -1.0], ID=0) # point 0
    g.point([4.0, -1.0], ID=1) # point 1
    g.point([4.0, 1.0], ID=2) # point 2
    g.point([0.0, 1.0], ID=3) # point 3

    g.spline([0, 1], ID=0) # line 0
    g.spline([1, 2], ID=1) # line 1
    g.spline([2, 3], ID=2) # line 2
    g.spline([3, 0], ID=3) # line 3

    g.surface([0, 1, 2, 3])

    mesh = cfm.GmshMesh(g)

    # Set the mesh parameters
    mesh.elType = 3 
    mesh.dofsPerNode = 2     
    mesh.elSizeFactor = 0.1

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()
    node_list, element_list = Mesh.create(coords, dofs, edof)

    #  The GUI saves the parameters to a file, load them
    parameters = load_config('config/parameters.json')

    # Create the system with the loaded parameters and geometry data
    system = System(node_list, element_list, parameters)

    # BC
    system.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
    system.load_point([4,-1],[0,-10])
    
    system.apply_dirichlet_bc()

    # Run the FEA
    system.solve_FE_sparse()

    # Visualize the results
    system.plot2(deformed=True)

if __name__ == "__main__":
    main()