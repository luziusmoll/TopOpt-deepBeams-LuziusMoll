import json
import sys
import os
import tkinter as tk
from tkinter import messagebox


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
    parser.add_argument('--no-gui', action='store_true',
                        help='Headless: skip all GUIs and plt.show(); read the existing JSON and only write the output PDFs')
    args = parser.parse_args()
    no_gui = args.no_gui

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
    if example_folder is None and not no_gui:
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
            print(f"No geometry file found at {geom_path}")
            return

    # Parameter input
    if example_folder is None and not no_gui:
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
            print(f"No parameter file found at {param_path}")
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

    # Run the optimization. dv = per-element area = dV/dx_e for the volume constraint
    # (matters on unstructured meshes; ~uniform for a regular gmsh grid).
    dv = system.sensitivity_densitiy()
    system.top_opt(dv, parameters['max_iteration'])

    # Always save to results/. If geometry comes from Examples, use results/<examplename>.png, otherwise use results/result.png
    # Always save as results/optimized_structure.pdf for STM GUI
    pdf_save_path = 'results/optimized_structure.pdf'
    system.plot_density(deformed=False, disp_bc=False, save_path=pdf_save_path, show=not no_gui)
    if example_folder is not None:
        example_pdf_path = os.path.join(example_folder, 'optimized_structure.pdf')
        system.plot_density(deformed=False, disp_bc=False, save_path=example_pdf_path, show=not no_gui)

    # If running from Examples, also save as optimized_structure.png in the example folder
    geom_path_norm = os.path.normpath(geom_path)
    parts = geom_path_norm.split(os.sep)
    # if 'Examples' in parts:
    #     idx = parts.index('Examples')
    #     if idx + 1 < len(parts):
    #         example_folder_path = os.path.join(*parts[:idx+2])
    #         png_save_path = os.path.join(example_folder_path, 'optimized_structure.png')
    #         system.plot_density(deformed=False, disp_bc=False, save_path=png_save_path)

    # GUI for strut and tie model
    if not no_gui:
        root = tk.Tk()
        truss_gui = TrussInputGUI(root, geometry_path=geom_path, parameter_path=param_path)
        root.mainloop()



if __name__ == "__main__":
    main()