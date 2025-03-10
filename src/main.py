import json
import sys
import os
from tkinter import Tk

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gui import ParameterInputGUI #, GeometryInputGUI
from utils.config import load_config
from system import System


def main():
    # Load parameters from JSON file
    parameters = load_config('config/parameters.json')

    # Initialize the GUI for geometry input
    root = Tk()
    gui = ParameterInputGUI(root)
    root.mainloop()

    # Assuming the GUI saves the parameters to a file, load them
    geometry_data = load_config('config/parameters.json')

    # Create the system with the loaded parameters and geometry data
    system = System(geometry_data, parameters)

    # Run the optimization
    system.solve_FE_sparse()

if __name__ == "__main__":
    main()