import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.io as pio
import cv2
from skimage.morphology import skeletonize
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan
import time
from shapely.geometry import Point, LineString


#%% image processing


def plot_image(image):
    """
    Plots an image with the shape (256, 256, 3).

    Parameters:
    - image: The image array to be plotted. It should be in the format (256, 256, 3).
              The function will ensure the image is in uint8 format before plotting.
    """
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    plt.imshow(image)
    plt.axis('off')  # Hide axis labels and ticks
    plt.show()
    

def save_image(image, file_path):
    folder_name = os.path.dirname(file_path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the image
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved preprocessed image as {file_path}")


#%% design space transformation from calfem to shapely 


def construct_polygons_from_neighbors(coords, bdofs):
    """
    Construct polygons by traversing boundary nodes.

    Parameters:
    - coords: Array of node coordinates (N x 2, where N is the number of nodes).
    - bdofs: Dictionary containing lists of DOFs for each boundary marker.

    Returns:
    - polygons: List of Shapely Polygons constructed from the boundary nodes.
    """
    if 0 not in bdofs:
        raise ValueError("No boundary DOFs found in the bdofs dictionary.")

    # Extract boundary nodes and coordinates
    boundary_nodes = np.unique(np.array(bdofs[0]) // 2)
    boundary_coords = coords[boundary_nodes]

    # Distance matrix
    dist = np.zeros((len(boundary_coords), len(boundary_coords)))
    for i in range(len(boundary_coords)):
        for j in range(len(boundary_coords)):
            dist[i, j] = np.sqrt((boundary_coords[i, 0] - boundary_coords[j, 0])**2 +
                                 (boundary_coords[i, 1] - boundary_coords[j, 1])**2)

    # Find the two closest neighbors for each node
    neighbors = {}
    for i in range(len(boundary_coords)):
        sorted_indices = np.argsort(dist[i])  # Sort distances for node `i`
        neighbors[i] = sorted_indices[1:3]   # Skip self (index 0) and take the two closest

    # Visited vector
    visited = np.zeros(len(boundary_coords), dtype=bool)
    polygons = []

    while not np.all(visited):
        # Start with the first unvisited node
        start_idx = np.where(~visited)[0][0]
        current_idx = start_idx
        polygon = []

        while True:
            # Mark current node as visited and add it to the polygon
            visited[current_idx] = True
            polygon.append(boundary_coords[current_idx])

            # Check neighbors
            unvisited_neighbors = [n for n in neighbors[current_idx] if not visited[n]]

            if len(unvisited_neighbors) == 0:
                # Both neighbors visited, polygon is closed
                break

            # Move to the first unvisited neighbor
            current_idx = unvisited_neighbors[0]

        # Close the polygon by connecting back to the start
        if len(polygon) >= 3:  # Ensure at least 3 unique points
            polygon.append(boundary_coords[start_idx])  # Close the loop
            polygons.append(Polygon(polygon))
        else:
            print(f"Skipped a component with fewer than 3 points: {polygon}")

    return polygons


def plot_polygons_and_nodes(coords, polygons):
    """
    Plot the nodes and the constructed polygons.

    Parameters:
    - coords: Array of node coordinates (N x 2, where N is the number of nodes).
    - polygons: List of Shapely Polygons to plot.

    Returns:
    None
    """
    plt.scatter(coords[:, 0], coords[:, 1], color='gray', label='All Nodes', alpha=0.5)

    for polygon in polygons:
        x, y = polygon.exterior.xy
        plt.plot(x, y, label='Polygon', color='blue')
        plt.scatter(x, y, color='red')  # Mark vertices

    plt.title("Polygons Constructed from Boundary Nodes")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_boundary_nodes(coords, bdofs):
    """
    Plot all nodes corresponding to the boundary degrees of freedom (DOFs).

    Parameters:
    - coords: Array of node coordinates (N x 2, where N is the number of nodes).
    - bdofs: Dictionary containing lists of DOFs for each boundary marker.

    Returns:
    None
    """
    if 0 not in bdofs:
        raise ValueError("No boundary DOFs found in the bdofs dictionary.")

    # Extract boundary nodes (assuming DOFs are indexed as [2n, 2n+1] for each node)
    boundary_nodes = np.unique(np.array(bdofs[0]) // 2)  # Integer division to map DOFs to nodes

    # Get coordinates of boundary nodes
    boundary_coords = coords[boundary_nodes]

    # Plot all nodes
    plt.scatter(coords[:, 0], coords[:, 1], color='gray', label='All Nodes', alpha=0.5)

    # Highlight boundary nodes
    plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], color='red', label='Boundary Nodes')

    # Add labels and styling
    plt.title("Boundary Nodes")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

