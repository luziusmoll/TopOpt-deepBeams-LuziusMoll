import tkinter as tk
from tkinter import messagebox
import json
from PIL import Image, ImageTk
import fitz  # PyMuPDF


class TrussInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Truss Input")

        self.load_points = []
        self.support_points = []

        self.load_geometry()

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.nodes = []
        self.trusses = []

        self.add_load_and_support_points()
        self.load_background()

        self.canvas.bind("<Button-1>", self.add_node)
        self.canvas.bind("<Button-3>", self.create_truss)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

    def load_geometry(self):
        with open('config/geometry.json', 'r') as f:
            data = json.load(f)
            self.load_points = data.get("load_points", [])
            self.support_points = data.get("support_points", [])

        # Calculate canvas dimensions based on min and max x and y values
        all_points = [point[0] for point in self.load_points] + self.support_points
        x_values = [point[0] for point in all_points]
        y_values = [point[1] for point in all_points]

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        width = max_x - min_x
        height = max_y - min_y

        # Scaling factor to fit the canvas within 800x800
        self.scale_factor = 800 / max(width, height)

        self.canvas_width = int(width * self.scale_factor)
        self.canvas_height = int(height * self.scale_factor)

        # Scale load and support points
        self.load_points = [[[x * self.scale_factor, y * self.scale_factor], load] for [[x, y], load] in self.load_points]
        self.support_points = [[x * self.scale_factor, y * self.scale_factor] for [x, y] in self.support_points]

    def add_load_and_support_points(self):
        # Add load and support points as nodes and show in GUI
        for point in self.load_points:
            self.nodes.append(point[0])
            self.canvas.create_oval(point[0][0]-10, point[0][1]-10, point[0][0]+10, point[0][1]+10, fill="green")
        for point in self.support_points:
            self.nodes.append(point)
            self.canvas.create_oval(point[0]-2, point[1]-2, point[0]+2, point[1]+2, fill="red")

    def load_background(self):
        try:
            # Convert PDF to image
            pdf_path = './results/optimized_structure.pdf'
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Resize image to fit canvas
            img = img.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            self.background_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background image: {e}")

    def add_node(self, event):
        x, y = event.x, event.y
        self.nodes.append((x, y))
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red")

    def create_truss(self, event):
        if len(self.nodes) < 2:
            messagebox.showerror("Input Error", "At least 2 nodes are required to create a truss.")
            return

        # Automatically create a truss between the last two nodes
        self.trusses.append((self.nodes[-2], self.nodes[-1]))
        self.canvas.create_line(self.nodes[-2], self.nodes[-1], fill="red")

    def submit(self):
        if not self.trusses:
            messagebox.showerror("Input Error", "No trusses created.")
            return

        print(f"Saving trusses with nodes: {self.nodes}")  # Debug print

        # Save the trusses and nodes to a config file
        data = {
            "nodes": self.nodes,
            "trusses": self.trusses
        }
        with open('config/trusses.json', 'w') as f:
            json.dump(data, f)

        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    gui = TrussInputGUI(root)
    root.mainloop()