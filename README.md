# TopOpt for deep beams

This project is a modular Python application designed for finite element analysis and optimization with a graphical user interface (GUI) for geometry input. The application allows users to define geometric shapes, set parameters for optimization, and visualize the results.

## Project Structure

```
TopOpt for deep beams
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── geometryGUI.py
│   ├── parameterGUI.py
│   ├── system.py
│   ├── system_setup.py
│   ├── beam_element.py
│   ├── membrane_element.py
│   ├── membrane.py
│   ├── node.py
│   ├── mesh.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   └── file_io.py
├── config
│   ├── parameters.json
│   └── geometry.json
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/luziusmoll/TopOpt-deepBeams-LuziusMoll.git
   cd TopOpt for deep beams
   ```

2. Create and activate a virtual environment 
   ```
   python3 -m venv .venv
   source .venv/bin/activate     # macOS / Linux
   .venv\Scripts\activate        # Windows
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage


You can run the application in two ways:

### 1. GUI Mode (default)

Launch the GUI to input geometry and parameters interactively:

```
python src/main.py
```

### 2. Example Folder Mode

Run a predefined example by specifying a folder containing `geometry.json` and `parameters.json`:

```
python src/main.py --example Examples/bridge_1
```

This will load the geometry and parameters from the specified folder and save the output PNG to the same folder. No GUI input is required in this mode.

## Configuration

Parameters are dumped from the GUI to  `config/parameters.json`. The following parameters can be chosen:
volfrac": "0.4", "penalty": "3", "x_min": "1e-3", "r_min": "4", "Youngs_modulus": "30000", "Poissons_ratio": "0.15", "max_iteration": "50", "mesh_el_size": "2"

Geometry and boundary conditions are dumped to  `config/geometry.json`. The geometry definition works as follows:
surfaces are defined as polygons by giving a list of nodes. Holes can in the first surface can be defined by adding more lists of Polygons, which define the holes.
Boundary conditions can be defined on nodes or lines of the polygons.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.