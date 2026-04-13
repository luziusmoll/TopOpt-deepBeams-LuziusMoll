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




import argparse

def main():

    parser = argparse.ArgumentParser(description="TopOpt Example Runner")
    parser.add_argument('--example', type=str, default=None, help='Path to example folder (containing geometry.json and parameters.json)')
    args = parser.parse_args()

    # Remove custom arguments so Gmsh doesn't see them
    import sys as _sys
    _sys.argv = [_sys.argv[0]]

    example_folder = args.example
    if example_folder is not None:
        example_folder = os.path.abspath(example_folder)
        geom_path = os.path.join(example_folder, 'geometry.json')
        param_path = os.path.join(example_folder, 'parameters.json')
    else:
        geom_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'geometry.json'))
        param_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'parameters.json'))

    # Geometry input
    if example_folder is None:
        try:
            resp = input("Do you want to input geometry data? [y/N]: ").strip().lower()
        except EOFError:
            print("No input available. Exiting.")
            return
        if resp in ('y', 'yes'):
            root = tk.Tk()
            geom_gui = GeometryInputGUI(root)
            root.mainloop()
            if hasattr(geom_gui, 'aborted') and geom_gui.aborted:
                print("Geometry input aborted by user. Exiting program.")
                return
        elif not os.path.exists(geom_path):
            print(f"No geometry file found at {geom_path}. Exiting.")
            return
    else:
        if not os.path.exists(geom_path):
            print(f"No geometry file found in example folder: {geom_path}")
            return

    # Parameter input
    if example_folder is None:
        try:
            resp = input("Do you want to input parameter data? [y/N]: ").strip().lower()
        except EOFError:
            print("No input available. Exiting.")
            return
        if resp in ('y', 'yes'):
            root = tk.Tk()
            param_gui = ParameterInputGUI(root)
            root.mainloop()
        elif not os.path.exists(param_path):
            print(f"No parameter file found at {param_path}. Exiting.")
            return
    else:
        if not os.path.exists(param_path):
            print(f"No parameter file found in example folder: {param_path}")
            return

    # Get node_list and element_list system setup
    system_setup = SystemSetup(geom_path)
    node_list, element_list = system_setup.create_mesh_from_geometry()

    if node_list is None or element_list is None:
        print("Error: node_list or element_list is None")
        return

    parameters = load_config(param_path)

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

    # Run the optimization
    dv = np.ones(len(system.elements))
    system.top_opt(dv, parameters['max_iteration'])

    # Save PNG to example folder if provided, else to results/
    if example_folder is not None:
        save_path = os.path.join(example_folder, 'optimized_structure.png')
    else:
        save_path = 'results/optimized_structure.png'
    system.plot2(deformed=False, disp_bc=False, save_path=save_path)

    # GUI for strut and tie model
    root = tk.Tk()
    truss_gui = TrussInputGUI(root)
    root.mainloop()


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