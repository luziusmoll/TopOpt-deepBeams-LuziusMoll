import json
import sys
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np


# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parameterGUI import ParameterInputGUI 
from src.geometryGUI import GeometryInputGUI
from src.strutandtieGUI import TrussInputGUI
from src.utils.config import load_config
from src.system import System
from src.system_setup import SystemSetup




def main():
    # Ask whether to proceed with geometry GUI input (terminal prompt)
    try:
        resp = input("Do you want to input geometry data? [y/N]: ").strip().lower()
    except EOFError:
        print("No input available. Exiting.")
        return
    if resp not in ('y', 'yes'):
        print("User chose not to proceed with geometry input.")
        # check if geometry.json exists
        geom_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'geometry.json'))
        if os.path.exists(geom_path):
            print(f"Found geometry file at {geom_path}; proceeding without GUI.")
        else:
            print("No 'config/geometry.json' found. Exiting.")
            return  # Exit the main function if user chooses not to proceed

    # Initialize the GUI for geometry input if the user wants to
    if resp in ('y', 'yes'):
        root = tk.Tk()
        geom_gui = GeometryInputGUI(root)
        root.mainloop()

        # Check if the operation was aborted
        if hasattr(geom_gui, 'aborted') and geom_gui.aborted:
            print("Geometry input aborted by user. Exiting program.")
            return  # Exit the main function and terminate the program
    # else:
    #     # User declined the GUI; allow proceeding if a geometry file exists
    #     geom_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'geometry.json'))
    #     if os.path.exists(geom_path):
    #         print(f"Using existing geometry file at {geom_path}. Proceeding without GUI.")
    #     else:
    #         print("No geometry input provided and no 'config/geometry.json' found. Exiting.")
    #         return
    
    # Ask whether to proceed with parameter GUI input (terminal prompt)
    try:
        resp = input("Do you want to input parameter data? [y/N]: ").strip().lower()
    except EOFError:
        print("No input available. Exiting.")
        return
    if resp not in ('y', 'yes'):
        print("User chose not to proceed with parameter input.")
        # check if parameters.json exists
        param_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'parameters.json'))
        if os.path.exists(param_path):
            print(f"Found parameter file at {param_path}; proceeding without GUI.")
        else:
            print("No 'config/parameters.json' found. Exiting.")
            return  # Exit the main function if user chooses not to proceed

    # Initialize the GUI for parameter input
    if resp in ('y', 'yes'):
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
    try:
        system = system_setup.apply_boundary_conditions(system)
    except Exception as e:
        print(f"Error in system definition: {e}")
        return

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
    system.top_opt(dv, parameters['max_iteration'])
    system.plot2(deformed=False, disp_bc=False, save_path='results/optimized_structure.pdf')

    # GUI for strut and tie model
    root = tk.Tk()
    truss_gui = TrussInputGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()