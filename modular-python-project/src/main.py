import json
import sys
from src.gui import GeometryInputGUI
from src.utils.config import load_parameters
from src.system import System

def main():
    # Load parameters from JSON file
    parameters = load_parameters('config/parameters.json')

    # Initialize the GUI for geometry input
    gui = GeometryInputGUI()
    geometry_data = gui.run()

    # Create the system with the loaded parameters and geometry data
    system = System(geometry_data, parameters)

    # Run the optimization
    system.solve_FE_sparse()

if __name__ == "__main__":
    main()