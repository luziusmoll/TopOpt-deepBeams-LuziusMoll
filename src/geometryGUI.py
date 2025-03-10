from tkinter import Tk, Label, Entry, Button, Frame, StringVar, messagebox
import json
import os

class GeometryInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Geometry Input")

        self.canvas = Tk.Canvas(master, width=400, height=400, bg="white")
        self.canvas.pack()

        self.points = []
        self.lines = []
        self.surfaces = []

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.create_surface)

        self.submit_button = Tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")

        if len(self.points) > 1:
            self.lines.append((self.points[-2], self.points[-1]))
            self.canvas.create_line(self.points[-2], self.points[-1])

    def create_surface(self, event):
        if len(self.points) < 3:
            messagebox.showerror("Input Error", "At least 3 points are required to create a surface.")
            return

        self.surfaces.append(self.points)
        self.points = []

    def submit(self):
        if not self.surfaces:
            messagebox.showerror("Input Error", "No surfaces created.")
            return

        g = cfg.Geometry()

        for surface in self.surfaces:
            point_ids = []
            for i, (x, y) in enumerate(surface):
                point_id = g.point([x, y], ID=i)
                point_ids.append(point_id)

            for i in range(len(point_ids)):
                g.spline([point_ids[i], point_ids[(i + 1) % len(point_ids)]], ID=i)

            g.surface(list(range(len(point_ids))))

        mesh = cfm.GmshMesh(g)
        mesh.elType = 3
        mesh.dofsPerNode = 2
        mesh.elSizeFactor = 0.1

        coords, edof, dofs, bdofs, elementmarkers = mesh.create()

        # Save the mesh data or pass it to the next step
        # For example, save to a file or pass to another function

        self.master.destroy()

def main():
    root = Tk()
    gui = GeometryInputGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()