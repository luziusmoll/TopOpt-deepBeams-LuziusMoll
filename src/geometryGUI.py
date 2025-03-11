import tkinter as tk
from tkinter import messagebox
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
import calfem.core as cfc
import matplotlib.pyplot as plt
from mesh import Mesh


class GeometryInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Geometry Input")

        self.canvas_width = 400
        self.canvas_height = 400
        self.grid_size = 20

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.draw_grid()
        self.draw_axes()

        self.points = []
        self.lines = []
        self.surfaces = []

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.create_surface)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

        self.node_list = None
        self.element_list = None

    def draw_grid(self):
        for i in range(0, self.canvas_width, self.grid_size):
            self.canvas.create_line([(i, 0), (i, self.canvas_height)], tag='grid_line', fill='lightgray')
        for i in range(0, self.canvas_height, self.grid_size):
            self.canvas.create_line([(0, i), (self.canvas_width, i)], tag='grid_line', fill='lightgray')

    def draw_axes(self):
        self.canvas.create_line(0, self.canvas_height / 2, self.canvas_width, self.canvas_height / 2, fill='black')
        self.canvas.create_line(self.canvas_width / 2, 0, self.canvas_width / 2, self.canvas_height, fill='black')

        for i in range(0, self.canvas_width, self.grid_size):
            self.canvas.create_text(i, self.canvas_height / 2 + 10, text=str(i - self.canvas_width / 2), fill='black')
        for i in range(0, self.canvas_height, self.grid_size):
            self.canvas.create_text(self.canvas_width / 2 + 10, i, text=str(self.canvas_height / 2 - i), fill='black')

    def snap_to_grid(self, x, y):
        x = round(x / self.grid_size) * self.grid_size
        y = round(y / self.grid_size) * self.grid_size
        return x, y

    def add_point(self, event):
        x, y = self.snap_to_grid(event.x, event.y)
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
            print(f"Creating surface with lines: {all_surfaces}")  # Debug print
            if len(all_surfaces) == 1:
                g.surface(all_surfaces[0], [])
            if len(all_surfaces) > 1:
                print(f"Creating surface with lines: {all_surfaces[0], all_surfaces[1:]}")  # Debug print
                g.surface(all_surfaces[0], all_surfaces[1:])
        except Exception as e:
            print(f"Exception occurred while creating surface: {e}")  # Debug print

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