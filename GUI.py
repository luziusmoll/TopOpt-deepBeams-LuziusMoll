import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import triangle as tr

class GeometryCreator:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.points = []
        self.lines = []
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.done_button = Button(plt.axes([0.8, 0.05, 0.1, 0.075]), 'Done')
        self.done_button.on_clicked(self.finish)
        self.finished = False

    def onclick(self, event):
        if event.inaxes == self.ax and not self.finished:
            self.points.append([event.xdata, event.ydata])
            if len(self.points) > 1:
                line, = self.ax.plot([self.points[-2][0], self.points[-1][0]], 
                                     [self.points[-2][1], self.points[-1][1]], 'k-')
                self.lines.append(line)
            self.ax.plot(event.xdata, event.ydata, 'ro')
            self.fig.canvas.draw()

    def finish(self, event):
        if len(self.points) > 2:
            line, = self.ax.plot([self.points[-1][0], self.points[0][0]], 
                                 [self.points[-1][1], self.points[0][1]], 'k-')
            self.lines.append(line)
            self.finished = True
            self.fig.canvas.mpl_disconnect(self.cid)
            self.done_button.ax.set_visible(False)
            self.fig.canvas.draw()
        else:
            print("At least three points are required to form a valid geometry.")

    def show(self):
        plt.show(block=True)
        return np.array(self.points)

def create_mesh_from_geometry(points, max_area=0.01):
    if len(points) < 3:
        raise ValueError("Input must have at least three vertices.")
    
    segments = [[i, (i+1) % len(points)] for i in range(len(points))]
    A = dict(vertices=points, segments=segments)
    B = tr.triangulate(A, 'pqa' + str(max_area))
    coords = B['vertices']
    triangles = B['triangles']
    return coords, triangles

# Use the GeometryCreator to define geometry
geometry_creator = GeometryCreator()
points = geometry_creator.show()

# Create the mesh from the defined geometry if there are enough points
if len(points) >= 3:
    coords, triangles = create_mesh_from_geometry(points)

    # Plot the mesh
    plt.figure()
    plt.triplot(coords[:, 0], coords[:, 1], triangles)
    plt.plot(coords[:, 0], coords[:, 1], 'o')
    plt.show()
else:
    print("Mesh generation aborted. At least three points are required.")
