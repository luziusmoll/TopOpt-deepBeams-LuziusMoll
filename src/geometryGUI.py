import tkinter as tk
from tkinter import messagebox, simpledialog
import json


class LoadVectorDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Load in x direction:").grid(row=0)
        tk.Label(master, text="Load in y direction:").grid(row=1)

        self.load_x = tk.Entry(master)
        self.load_y = tk.Entry(master)

        self.load_x.grid(row=0, column=1)
        self.load_y.grid(row=1, column=1)
        return self.load_x  # initial focus

    def apply(self):
        try:
            self.result = [float(self.load_x.get()), float(self.load_y.get())]
        except ValueError:
            self.result = None


class GeometryInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Geometry Input")

        # Initialize abort flag
        self.aborted = False

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

        self.load_points = []
        self.load_lines = []
        self.support_points = []
        self.support_lines = []

        self.mode = "geometry"
        
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.create_surface)

        self.master.bind("l", self.set_load_mode)
        self.master.bind("s", self.set_support_mode)
        self.master.bind("<Escape>", self.set_geometry_mode)


        # Create a frame for buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        self.submit_button = tk.Button(self.button_frame, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.LEFT, padx=5)

        self.abort_button = tk.Button(self.button_frame, text="Abort", command=self.abort)
        self.abort_button.pack(side=tk.LEFT, padx=5)

        # Add a legend
        self.legend = tk.Label(master, text="Left click to add points to Polygon, Right click to close current polygon\n" \
            "Press 'l' to switch to load definition\n 's' to switch to support definition\n 'Esc' to switch to geometry mode")
        self.legend.pack()


    def draw_grid(self):
        for i in range(0, self.canvas_width, self.grid_size):
            self.canvas.create_line([(i, 0), (i, self.canvas_height)], tag='grid_line', fill='lightgray')
        for i in range(0, self.canvas_height, self.grid_size):
            self.canvas.create_line([(0, i), (self.canvas_width, i)], tag='grid_line', fill='lightgray')

    def draw_axes(self):
        # Draw bounding box as axes
        self.canvas.create_line(0, 0, self.canvas_width, 0, fill='black')  # Top border
        self.canvas.create_line(0, 0, 0, self.canvas_height, fill='black')  # Left border
        self.canvas.create_line(self.canvas_width, 0, self.canvas_width, self.canvas_height, fill='black')  # Right border
        self.canvas.create_line(0, self.canvas_height, self.canvas_width, self.canvas_height, fill='black')  # Bottom border

        # X-axis labels (bottom)
        for i in range(0, self.canvas_width, self.grid_size):
            self.canvas.create_text(i, self.canvas_height - 10, text=str(i), fill='black')  

        # Y-axis labels (left, corrected)
        for i in range(0, self.canvas_height, self.grid_size):
            corrected_y = (self.canvas_height - i)  # Flip the Y values to match Cartesian convention
            self.canvas.create_text(10, i, text=str(corrected_y), fill='black')
    
    def snap_to_grid(self, x, y):
        x = round(x / self.grid_size) * self.grid_size
        y = round(y / self.grid_size) * self.grid_size
        return x, y

    def add_point(self, event):
        x, y = self.snap_to_grid(event.x, event.y)
        if self.mode == "geometry":
            self.points.append((x, y))
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")

            if len(self.points) > 1:
                self.lines.append((self.points[-2], self.points[-1]))
                self.canvas.create_line(self.points[-2], self.points[-1])
        elif self.mode == "load":
            load_vector = self.get_load_vector()
            if load_vector:
                self.load_points.append([(x, y), load_vector])
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="green")
        elif self.mode == "support":
            self.support_points.append((x, y))
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")

    def get_load_vector(self):
        dialog = LoadVectorDialog(self.master)
        return dialog.result
    
    def create_surface(self, event):
        if len(self.points) < 3:
            messagebox.showerror("Input Error", "At least 3 points are required to create a surface.")
            return

        # Automatically close the surface by connecting the last point to the first point
        self.lines.append((self.points[-1], self.points[0]))
        self.canvas.create_line(self.points[-1], self.points[0])

        self.surfaces.append(self.points)
        self.points = []

    def set_load_mode(self, event):
        self.mode = "load"

    def set_support_mode(self, event):
        self.mode = "support"

    def set_geometry_mode(self, event):
        self.mode = "geometry"

    def abort(self):
        # Set a flag to indicate the operation was aborted
        self.aborted = True
        self.master.destroy()

    def submit(self):
        if not self.surfaces:
            messagebox.showerror("Input Error", "No surfaces created.")
            return

        print(f"Saving geometry with surfaces: {self.surfaces}")  # Debug print

        # Convert screen coordinates to Cartesian coordinates (flip y)
        cartesian_surfaces = [[(x, self.canvas_height - y) for x, y in surface] for surface in self.surfaces]
        cartesian_load_points = [[(x, self.canvas_height - y), load] for [(x, y), load] in self.load_points]
        cartesian_support_points = [(x, self.canvas_height - y) for x, y in self.support_points]
        
        # Lines need both points converted
        cartesian_load_lines = [[(x1, self.canvas_height - y1), (x2, self.canvas_height - y2)] 
                              for [(x1, y1), (x2, y2)] in self.load_lines]
        cartesian_support_lines = [[(x1, self.canvas_height - y1), (x2, self.canvas_height - y2)]
                                 for [(x1, y1), (x2, y2)] in self.support_lines]

        # dump the geometry and loads and supports to the config file
        data = {
            "surfaces": cartesian_surfaces,
            "load_points": cartesian_load_points,
            "load_lines": cartesian_load_lines,
            "support_points": cartesian_support_points,
            "support_lines": cartesian_support_lines
        }
        with open('config/geometry.json', 'w') as f:
            json.dump(data, f)

        self.master.destroy()
