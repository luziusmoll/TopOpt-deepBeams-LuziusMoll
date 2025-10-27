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
   .venv\Scripts\activate      # Windows
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command in the terminal:

```
python src/main.py
```

This will launch the GUI, allowing you to input geometry and parameters for the finite element model.

## Configuration

Parameters are dumped from the GUI to  `config/parameters.json`. Geometry and boundary conditions are dumped to  `config/geometry.json`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.