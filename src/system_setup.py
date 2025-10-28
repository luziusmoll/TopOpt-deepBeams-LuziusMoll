import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config
from src.mesh import Mesh


class SystemSetup:
    # def __init__(self):
    #     self.surfaces = None  


    # def create_geometry(self):
    #     # read the surfaces from the json file
    #     with open('config/geometry.json', 'r') as f:
    #         self.surfaces = json.load(f)
    #         print(f"Read surfaces from file: {self.surfaces}")  # Debug print
    def __init__(self):
        self.surfaces = None
        self.load_points = None
        self.load_lines = None
        self.support_points = None
        self.support_lines = None

    def create_mesh_from_geometry(self):
        # read the geometry, loads, and supports from the json file
        with open('config/geometry.json', 'r') as f:
            data = json.load(f)
            self.surfaces = data.get("surfaces", [])
            self.load_points = data.get("load_points", [])
            self.load_lines = data.get("load_lines", [])
            self.support_points = data.get("support_points", [])
            self.support_lines = data.get("support_lines", [])
            print(f"Read data from file: {data}")  # Debug print
        g = cfg.Geometry()

        pID = 0  # pID for all points, num_points for each surface
        sID = 0  # sID for all splines
        all_surfaces = []
        for surface in self.surfaces:
            print(f"Creating surface with points: {surface}")  # Debug print
            for i, (x, y) in enumerate(surface):
                if x is None or y is None:
                    print(f"Skipping invalid point: ({x}, {y})")  # Debug print
                    continue
                print(f"Adding point: ({x, y}), ID={pID}")  # Debug print
                g.point([x, y], ID=pID)
                num_points = i
                pID += 1

            if num_points < 2:
                print(f"Skipping surface creation due to insufficient points: {num_points}")  # Debug print
                continue

            for i in range(num_points):
                print(f"Adding spline: ({sID}, {(sID + 1)}), ID={sID}")  # Debug print
                try:
                    g.spline([sID, (sID + 1)], ID=sID)
                except Exception as e:
                    print(f"Exception occurred while adding spline ({sID}, {(sID + 1)}): {e}")  # Debug print
                    continue
                sID += 1

            # close the surface
            try:
                print(f"Adding spline: ({sID}, {sID-num_points}), ID={sID}")  # Debug print
                g.spline([sID, sID-num_points], ID=sID)
                sID += 1
            except Exception as e:
                print(f"Exception occurred while adding spline ({sID}, {sID-num_points}): {e}")  # Debug print
                continue

            # print(f"Creating surface with points: {len(self.points)}")  # Debug print
            all_surfaces.append(list(range(sID-num_points-1, sID)))

        try:
            if len(all_surfaces) == 1:
                print(f"Creating surface with lines: {all_surfaces}")  # Debug print
                g.surface(all_surfaces[0], [])
            if len(all_surfaces) > 1:
                print(f"Creating surface with lines: {all_surfaces[0], all_surfaces[1:]}")  # Debug print
                g.surface(all_surfaces[0], all_surfaces[1:])
        except Exception as e:
            print(f"Exception occurred while creating surface: {e}")  # Debug print

        # cfv.drawGeometry(g)
        # cfv.showAndWait()

        mesh = cfm.GmshMesh(g)
        mesh.elType = 3
        mesh.dofsPerNode = 2

        # Element size factor
        parameters = load_config('config/parameters.json')
        mesh.elSizeFactor = parameters.get("mesh_el_size")
        #mesh.elSizeFactor = 10

        try:
            print("Creating mesh...")  # Debug print
            coords, edof, dofs, bdofs, elementmarkers = mesh.create()
            node_list, element_list = Mesh.create(coords, dofs, edof)
            return node_list, element_list
        except Exception as e:
            print(f"Exception occurred while creating mesh: {e}")  # Debug print
    

    def apply_boundary_conditions(self, system):
        # Check for sufficient supports (at least 2)
        total_supports = len(self.support_points) + 2 * len(self.support_lines)
        if total_supports < 2:
            raise RuntimeError("System is underspecified: At least 2 support points/1 line are required for stability.")
            
        # Check for existence of loads
        total_loads = len(self.load_points) + len(self.load_lines)
        if total_loads == 0:
            raise RuntimeError("System is underspecified: No loads have been defined.")

        # Apply supports
        for p in self.support_points:
            system.fix_node_by_coord(p)
        
        for l in self.support_lines:
            system.fix_line(l[0], l[1])     

        # Apply loads
        for p in self.load_points:
            system.load_point(p[0], p[1])
        
        for l in self.load_lines:
            system.load_line(l[0], l[1])

        system.apply_dirichlet_bc()

        return system