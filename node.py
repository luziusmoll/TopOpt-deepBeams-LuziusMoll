import numpy as np


class Node:
    def __init__(self, coords, id, dofs, fixed = [False,False], forces = np.zeros(2)) -> None:
        self.coords = coords
        self.id = id
        self.dofs = dofs
        self.forces = forces
        self.fixed = fixed
        self.displacements = np.zeros(2)

    def current_coords(self):
        return self.coords + self.displacements
 