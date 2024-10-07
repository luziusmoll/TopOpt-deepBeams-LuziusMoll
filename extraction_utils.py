from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import cv2


def is_black(image, x, y, threshold=50):
    """
    Checks if the pixel at (x, y) is black based on a grayscale threshold.

    Parameters:
    - image: The input grayscale image.
    - x, y: The coordinates of the pixel to check.
    - threshold: The intensity value below which a pixel is considered black (default is 50).

    Returns:
    - True if the pixel is black, False otherwise.
    """
    return image[y, x] < threshold


def is_mostly_black(image, center_x, center_y, radius, lower_threshold=0.0, upper_threshold=0.9):
    """
    Checks if the percentage of black pixels within a circle around (center_x, center_y)
    is between the specified lower and upper thresholds.

    Parameters:
    - image: The input grayscale image.
    - center_x, center_y: The coordinates of the center of the circle.
    - radius: The radius of the circle.
    - lower_threshold: The lower bound of the percentage of black pixels.
    - upper_threshold: The upper bound of the percentage of black pixels.

    Returns:
    - True if the percentage of black pixels within the circle is between the thresholds, False otherwise.
    """
    black_pixel_count = 0
    total_pixel_count = 0
    
    for theta in np.linspace(0, 2 * np.pi, 360):  # Iterate over angles to cover the circle's boundary
        for r in range(radius + 1):  # Iterate from the center out to the radius
            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
                total_pixel_count += 1
                if image[y, x] == 0:  # Black pixel
                    black_pixel_count += 1

    black_ratio = black_pixel_count / total_pixel_count
    
    return lower_threshold <= black_ratio <= upper_threshold



def merge_segments(segments, min_angle_diff):
    """
    Merge consecutive segments if the angle difference between them is smaller than min_angle_diff.

    Parameters:
    - segments: List of tuples representing (start_angle, end_angle) for each segment.
    - min_angle_diff: The minimum angle difference (in radians) required to keep segments separate.

    Returns:
    - merged_segments: A list of merged segments.
    """
    
    if not segments:
        return segments

    merged_segments = [segments[0]]  # Start with the first segment

    for i in range(1, len(segments)):
        prev_start, prev_end = merged_segments[-1]
        curr_start, curr_end = segments[i]

        # If the angle difference between the previous segment's end and the current segment's start is small, merge them
        if curr_start - prev_end < min_angle_diff:
            merged_segments[-1] = (prev_start, curr_end)
        else:
            merged_segments.append((curr_start, curr_end))

    # Handle wrapping around at 0° (i.e., 2π)
    if len(merged_segments) > 1 and (2 * np.pi - merged_segments[-1][1] + merged_segments[0][0]) < min_angle_diff:
        # Merge the last segment with the first one
        merged_segments[0] = (merged_segments[-1][0], merged_segments[0][1])
        merged_segments.pop()  # Remove the last segment since it's merged with the first one


    return merged_segments


def check_cone_for_white_pixels(image, center, radius, start_angle, end_angle, white_threshold=0.05): #0.05
    """
    Checks if the cone defined by the center of the circle and the segment contains more than
    a specified percentage of white pixels.

    Parameters:
    - image: The input grayscale image.
    - center: The (x, y) coordinates of the circle center.
    - radius: The radius of the circle.
    - start_angle: The starting angle of the segment (in radians).
    - end_angle: The ending angle of the segment (in radians).
    - white_threshold: The percentage threshold of white pixels (default is 20%).

    Returns:
    - True if more than the specified threshold of pixels are white, False otherwise.
    """
    x_center, y_center = center
    total_pixels = 0
    white_pixels = 0

    # Adjust for circular wrapping: if end_angle is smaller than start_angle, add 2π to end_angle
    if end_angle < start_angle:
        end_angle += 2 * np.pi

    # Iterate over the angles in the segment's range
    for theta in np.linspace(start_angle, end_angle, int((end_angle - start_angle) * 180 / np.pi)):
        for r in range(1, radius + 1):  # Check every pixel in the radial direction
            x_circ = int(x_center + r * np.cos(theta))
            y_circ = int(y_center + r * np.sin(theta))
            
            if 0 <= x_circ < image.shape[1] and 0 <= y_circ < image.shape[0]:  # Check bounds
                total_pixels += 1
                if image[y_circ, x_circ] == 255:  # White pixel
                    white_pixels += 1
    
    # Calculate the white pixel ratio
    if total_pixels == 0:  # Avoid division by zero
        return False

    white_ratio = white_pixels / total_pixels

    # Return True if more than the threshold of pixels are white
    return white_ratio > white_threshold



def filter_segments_by_cone(image, center, radius, segments, white_threshold=0.05):
    """
    Filters out segments where more than the specified percentage of pixels in the cone are white.

    Parameters:
    - image: The input grayscale image.
    - center: The (x, y) coordinates of the circle center.
    - radius: The radius of the circle.
    - segments: A list of (start_angle, end_angle) tuples representing the segments.
    - white_threshold: The percentage threshold of white pixels to filter segments.

    Returns:
    - filtered_segments: A list of filtered segments.
    """
    filtered_segments = []

    for start_angle, end_angle in segments:
        if not check_cone_for_white_pixels(image, center, radius, start_angle, end_angle, white_threshold):
            filtered_segments.append((start_angle, end_angle))

    return filtered_segments


def find_node_candidates(image, radius=10, min_angle_diff=np.deg2rad(25), white_threshold=0.05):
    """
    Finds node candidates based on a fixed radius and neighborhood check, with additional filtering for white pixels in cones.

    Parameters:
    - image: The input grayscale image.
    - radius: The fixed radius for the circle.
    - white_threshold: The percentage threshold of white pixels to filter segments.

    Returns:
    - node_candidates: A list of (x, y) tuples representing the node positions.
    - segments_info: A dictionary containing the start and end angles for each segment for each node.
    """
    node_candidates = []
    segments_info = {}

    # Ensure the image is in grayscale format
    if len(image.shape) == 3:  # If the image has 3 channels (e.g., RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    rows, cols = image.shape  # Ensure this is 2D

    for x in range(radius, cols - radius):
        for y in range(radius, rows - radius):
            if is_black(image, x, y):
                if is_mostly_black(image, x, y, radius=radius):
                    # Check circumference
                    segments = []
                    segment_start = None
                    for theta in np.linspace(0, 2 * np.pi, 360):
                        x_circ = int(x + radius * np.cos(theta))
                        y_circ = int(y + radius * np.sin(theta))
                        
                        if is_black(image, x_circ, y_circ):
                            if segment_start is None:  # Start of a new segment
                                segment_start = theta
                        else:
                            if segment_start is not None:  # End of the current segment
                                segments.append((segment_start, theta))
                                segment_start = None  # Reset for the next segment
                    
                    # After the loop, check if the last segment closed the circle
                    if segment_start is not None:
                        segments.append((segment_start, 2 * np.pi))
    
                    # Merge segments with small angle differences
                    merged_segments = merge_segments(segments, min_angle_diff)
    
                    # Filter segments by checking the cone for white pixels
                    filtered_segments = filter_segments_by_cone(image, (x, y), radius, merged_segments, white_threshold)
    
                    # Classify the center as a node based on the number of filtered segments
                    if classify_node_by_segments(filtered_segments):
                        node_candidates.append((x, y))
                        segments_info[(x, y)] = filtered_segments
                        
    print(len(node_candidates), 'node candidates found')
    radii = np.ones(len(node_candidates))*radius
    return node_candidates, segments_info, radii


def classify_node_by_segments(segments):
    """
    Classifies a center point as a node based on the number of segments detected.
    """
    # Check if there are more than 2 segments
    if len(segments) > 2:
        return True
    
    # If there are 2 or fewer segments, it's not a node
    return False


def plot_node_with_segments(image, node, radius, segments):
    """
    Plot a circle around the node and visualize the detected segments.
    
    Parameters:
    - image: The input grayscale image.
    - node: The coordinates of the node (x, y).
    - radius: The radius used to detect segments.
    - segments: A list of tuples (start_angle, end_angle) representing the segments.
    """
    x, y = node

    # Plot the original image
    plt.imshow(image, cmap='gray')

    # Draw the circle around the node
    circle = plt.Circle((x, y), radius, color='blue', fill=False, linewidth=2)
    plt.gca().add_patch(circle)

    # Plot the segments on the circle
    for start_angle, end_angle in segments:
        # Handle wrapping of the end_angle
        if end_angle < start_angle:
            end_angle += 2 * np.pi

        angles = np.linspace(start_angle, end_angle, int((end_angle - start_angle) * 180 / np.pi))
        for angle in angles:
            x_circ = x + radius * np.cos(angle)
            y_circ = y + radius * np.sin(angle)
            plt.plot(x_circ, y_circ, 'ro', markersize=1)  # Mark segment points as red

    plt.scatter(x, y, color='green', s=5)  # Mark the node center as green
    plt.axis('off')
    plt.show()


def plot_all_nodes(image, node_candidates):
    """
    Plots all detected node candidates on the image.
    
    Parameters:
    - image: The input grayscale image.
    - node_candidates: A list of (x, y) tuples representing the node positions.
    """
    print('plotting nodes')
    plt.imshow(image, cmap='gray')

    # Plot all nodes as green points
    for (x, y) in node_candidates:
        plt.scatter(x, y, color='green', s=1)

    plt.axis('off')
    plt.show()


def cluster_nodes(node_candidates, eps=5, min_samples=2):
    """
    Cluster the node candidates using DBSCAN and find the centers of the clusters.

    Parameters:
    - node_candidates: A list of (x, y) tuples representing the node positions.
    - eps: The maximum distance between two points for them to be considered as in the same neighborhood.
    - min_samples: The minimum number of points required to form a dense region (cluster).

    Returns:
    - cluster_centers: A list of (x, y) tuples representing the center of each cluster.
    - labels: The cluster labels for each node candidate.
    """
    # Convert node_candidates to a NumPy array
    node_candidates_array = np.array(node_candidates)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(node_candidates_array)
    
    # Get the cluster labels (-1 means the point is considered noise)
    labels = clustering.labels_
    
    # Find the centers of the clusters
    cluster_centers = []
    for cluster_id in set(labels):
        if cluster_id != -1:  # Ignore noise points
            # Get all points that belong to this cluster
            cluster_points = node_candidates_array[labels == cluster_id]
            # Calculate the center of the cluster
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(tuple(cluster_center))
    
    return cluster_centers, labels


def plot_cluster_centers(image, cluster_centers, label='nodes'):
    """
    Plots the cluster centers on the image.

    Parameters:
    - image: The input grayscale image.
    - cluster_centers: A list of (x, y) tuples representing the cluster centers.
    """
    plt.imshow(image, cmap='gray')

    # Plot all cluster centers as blue 'x' points
    for (x, y) in cluster_centers:
        plt.scatter(x, y, color='blue', label=label, marker='x', s=100)

    # Ensure only unique entries in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removes duplicate labels by using a dictionary

    # Set plot settings
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.title("Extracted Nodes")
    plt.show()

    
    
    
#%% Xia 2020a


def thinning_iteration(img, iter_num):
    # Convert the image to binary format if it’s not already
    img = img // 255  # Convert to 0 and 1 for logical operations
    
    rows, cols = img.shape
    marker = np.zeros_like(img, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            P2 = img[i - 1, j]
            P3 = img[i - 1, j + 1]
            P4 = img[i, j + 1]
            P5 = img[i + 1, j + 1]
            P6 = img[i + 1, j]
            P7 = img[i + 1, j - 1]
            P8 = img[i, j - 1]
            P9 = img[i - 1, j - 1]
            P1 = img[i, j]

            # Number of non-zero neighbors
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            
            # Number of transitions from 0 to 1 in the neighborhood
            A = 0
            if P2 == 0 and P3 == 1:
                A+=1
            if P3 == 0 and P4 == 1:
                A+=1
            if P4 == 0 and P5 == 1:
                A+=1
            if P5 == 0 and P6 == 1:
                A+=1
            if P6 == 0 and P7 == 1:
                A+=1
            if P7 == 0 and P8 == 1:
                A+=1
            if P8 == 0 and P9 == 1:
                A+=1
            if P9 == 0 and P2 == 1:
                A+=1

            # Thinning conditions
            if iter_num == 0:
                C1 = P2 * P4 * P6 == 0
                C2 = P4 * P6 * P8 == 0
            else:
                C1 = P2 * P4 * P8 == 0
                C2 = P2 * P6 * P8 == 0

            if (P1 == 1) and (2 <= B <= 6) and (A == 1) and C1 and C2:
                marker[i, j] = 1

    img[marker == 1] = 0  # Remove the marked pixels
    
    return img * 255  # Convert back to 0 and 255

def convert_to_binary(img):
    """Converts a grayscale or RGB image to binary (0 and 255) in uint8 format."""
    # Convert RGB to grayscale if the image has multiple channels
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)
    
    # Convert grayscale to binary (0 or 255)
    binary_img = np.where(img > 128, 255, 0).astype(np.uint8)
    
    return binary_img

def zhang_suen_thinning(img):
    print('thinning started')
    img = convert_to_binary(img)  # Ensure the image is binary
    prev_img = np.zeros_like(img)
    iteration = 0
    while not np.array_equal(img, prev_img):
        prev_img = img.copy()
        # subiteration 0
        img = thinning_iteration(img, 0)
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Thinning Iteration {iteration} subiteration 0")
        # plt.show()
        # subiteration 1
        img = thinning_iteration(img, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Thinning Iteration {iteration} subiteration 1")
        # plt.show()
        iteration += 1
        
    print('thinning finished after ', iteration,' iterations')
    return img


# Function to check if a 3x3 block contains a pattern (pattern must be contained within the block)
def contains_pattern(block, pattern):
    return np.all((pattern == 0) | (block == pattern))

# Function to check if a 3x3 block matches any pattern (including rotations)
def matches_pattern(block):
    # Define the node patterns as binary 3x3 matrices
    patterns = [
        np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]]),  # Pattern 1 (rotational equivalents not shown)
        np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 2
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),  # Pattern 3
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 4
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])   # Pattern 5
    ]
    for pattern in patterns:
        # Check all four rotations (0, 90, 180, 270 degrees)
        for _ in range(4):
            if contains_pattern(block, pattern):
                return True
            # Rotate pattern 90 degrees
            pattern = np.rot90(pattern)
    return False

# Sliding window over the skeletonized image
def detect_nodes(skeletonized_image):
    rows, cols = skeletonized_image.shape
    node_positions = []

    # Slide a 3x3 window over the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            block = skeletonized_image[i-1:i+2, j-1:j+2]  # Extract 3x3 block
            if matches_pattern(block):
                node_positions.append((j, i))  # Add node position if pattern matches

    return node_positions

#%% path following

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from collections import deque


def follow_skeleton_path_bfs(start_node, skeleton_img, visited, nodes):
    """
    Follows the skeleton path starting from a node using BFS and stops 
    following a path once another node is detected, but continues exploring other paths.
    """
    cx, cy = map(int, start_node)  # Start node as integers
    connections = []
    
    # Mark the current node as visited
    visited[cx, cy] = 1
    
    # Create a copy of the skeleton image for debugging visualization
    debug_img = skeleton_img.copy()
    
    # Initialize the queue for BFS
    queue = [(cx, cy)]
    
    while queue:
        cx, cy = queue.pop(0)  # Dequeue the first element (BFS)
        
        # Get neighbors of the current point
        neighbors = get_neighbors_debug(cx, cy, skeleton_img)
        
        for nx, ny in neighbors:
            nx, ny = int(nx), int(ny)  # Ensure coordinates are integers
            
            # If the neighbor is another node and not the starting node
            if visited[nx, ny] == 2 and (nx, ny) != (int(start_node[0]), int(start_node[1])):
                connections.append((start_node, (nx, ny)))  # Save the connection
                
                # # Debug visualization (optional)
                # debug_img[ny, nx] = 127  # Mark path in gray
                # debug_img_uint8 = (debug_img * 255).astype(np.uint8)
                # debug_img_colored = cv2.cvtColor(debug_img_uint8, cv2.COLOR_GRAY2RGB)
                # cx_int, cy_int = tuple(map(int, (cx, cy)))
                # nx_int, ny_int = tuple(map(int, (nx, ny)))
                # cv2.circle(debug_img_colored, (nx_int, ny_int), 3, (0, 255, 0), -1)  # Mark endpoint in green
                # cv2.line(debug_img_colored, (int(start_node[0]), int(start_node[1])), (nx_int, ny_int), (255, 0, 0), 2)
                
                # # Display the debug image (intermediate steps)
                # plt.imshow(debug_img_colored)
                # plt.title(f" Connection from ({int(start_node[0])}, {int(start_node[1])}) to ({nx_int}, {ny_int})")
                # plt.axis('off')
                # plt.show()
                
                # Remove all pixels in a 5x5 grid around the found node from the queue to make sure the path is no longer followed
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        grid_x, grid_y = nx + dx, ny + dy
                        if (grid_x, grid_y) in queue:
                            queue.remove((grid_x, grid_y))
                
                # Stop following this path by skipping the current neighbor (nx, ny)
                break
            

            # If the neighbor is part of the skeleton and unvisited
            elif visited[nx, ny] == 0:
                visited[nx, ny] = 1  # Mark as visited
                queue.append((nx, ny))  # Add to queue to continue BFS search
                
                # # Debug visualization of intermediate steps
                # debug_img[ny, nx] = 127  # Mark path in gray
                # debug_img_uint8 = (debug_img * 255).astype(np.uint8)
                # debug_img_colored = cv2.cvtColor(debug_img_uint8, cv2.COLOR_GRAY2RGB)
                # cx_int, cy_int = tuple(map(int, (cx, cy)))
                # nx_int, ny_int = tuple(map(int, (nx, ny)))
                # cv2.circle(debug_img_colored, (nx_int, ny_int), 3, (0, 255, 0), -1)  # Mark current pixel
                # cv2.line(debug_img_colored, (int(start_node[0]), int(start_node[1])), (nx_int, ny_int), (255, 0, 0), 2)
                
                # plt.imshow(debug_img_colored)
                # plt.title(f"Following path from ({int(start_node[0])}, {int(start_node[1])}) to ({nx_int}, {ny_int})")
                # plt.axis('off')
                # plt.show()
    
    return connections


def get_neighbors_debug(x, y, skeleton_img):
    """
    Returns the 8-connected neighbors of the pixel (x, y) that are part of the skeleton (pixel value = 1).

    Parameters:
    - x, y: Coordinates of the current pixel.
    - skeleton_img: Binary skeletonized image (1 for skeleton, 0 for background).

    Returns:
    - neighbors: List of neighboring coordinates that are part of the skeleton.
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the current point
            nx, ny = int(x + dx), int(y + dy)
            if 0 <= nx < skeleton_img.shape[1] and 0 <= ny < skeleton_img.shape[0]:
                if skeleton_img[ny, nx] == 1:  # Check if it's part of the skeleton
                    neighbors.append((nx, ny))
    return neighbors


def generate_truss_structure_bfs_debug(nodes, skeleton_img):
    """
    Generates a truss-like structure by detecting straight-line connections between nodes along the skeleton.
    This version uses BFS to explore all possible paths and visualize the path-following process.
    
    Parameters:
    - nodes: List of node coordinates
    - skeleton_img: Binary skeletonized image (1 for skeleton pixels, 0 for background)
    
    Returns:
    - List of connections [(start_node, end_node)] that represent straight trusses.
    """
    truss_connections = []
    
    for node in nodes:
        
        # Initialize visited matrix for each node
        visited = np.zeros(skeleton_img.shape, dtype=int)
        
        # Mark all node locations in the visited matrix as "2" (special value for nodes)
        for node_center in nodes:
            # Round the node coordinates to the nearest integer
            nx, ny = round(node_center[0]), round(node_center[1])
            
            # Ensure the coordinates are within the bounds of the skeleton image
            if 0 <= nx < skeleton_img.shape[1] and 0 <= ny < skeleton_img.shape[0]:
                visited[nx, ny] = 2  # Mark this node as a special value
        
        # Find connections starting from the current node using BFS
        connections = follow_skeleton_path_bfs(node, skeleton_img, visited, nodes)
        truss_connections.extend(connections)
    
    return truss_connections


def plot_truss_structure(image, truss_connections, nodes):
    """
    Visualizes the truss structure by drawing straight lines between connected nodes.
    """
    img_copy = image.copy()
    for (start, end) in truss_connections:
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        cv2.line(img_copy, start, end, (0, 255, 0), 2)  # Green lines for trusses
    
    # Plot the nodes as blue circles
    for node in nodes:
        node = tuple(map(int, node))
        cv2.circle(img_copy, node, 5, (255, 0, 0), -1)  # Blue nodes
    
    # Display the image with the truss structure
    plt.figure(figsize=(10, 10))
    plt.imshow(img_copy)
    plt.title("Truss-like Structure with Nodes and Bars")
    plt.axis('off')
    plt.show()

# for the boundary conditions find the nearest part of the skeleton
def find_bcs_in_skeleton(skeletonized_image, nodes_stm_bc_img):
    """
    Finds the nearest skeleton points (black pixels) for each boundary condition node using a circular search pattern.
    
    Parameters:
    - skeletonized_image: The binary skeleton image (1 for skeleton pixels, 0 for background).
    - nodes_stm_bc_img: List of boundary condition nodes in the image space.
    
    Returns:
    - nodes_skel_bc: List of coordinates of the nearest skeleton pixels for each boundary condition node.
    """
    nodes_skel_bc = []

    for node in nodes_stm_bc_img:
        cx, cy = map(int, node)  # Node coordinates as integers
        
        detected = False
        distance = 1  # Start with a small radius
        
        while not detected:
            # Circular search: iterate over a grid of points within a square, but only keep points within the circle radius
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    # Check if the point is within the current radius (circular search)
                    if np.sqrt(dx**2 + dy**2) <= distance:
                        nx, ny = cx + dx, cy + dy  # Neighbor coordinates

                        # Ensure the coordinates are within the image bounds
                        if 0 <= nx < skeletonized_image.shape[1] and 0 <= ny < skeletonized_image.shape[0]:
                            # Check if the neighbor is a black pixel (skeleton pixel = 1)
                            if skeletonized_image[ny, nx] == 1:
                                nodes_skel_bc.append((nx, ny))  # Append the coordinates of the black pixel
                                detected = True
                                break

                if detected:
                    break
            
            distance += 1  # Increment the search radius if no pixel was found in the current range

    return nodes_skel_bc

#%% computer vision 

# Function to detect the intersection of two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return int(px), int(py)
    return None

# Function to calculate the angle between two lines
def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    angle = np.abs(angle1 - angle2)
    if angle > np.pi:
        angle = 2 * np.pi - angle
    return angle * 180 / np.pi

def line_detection_plot(binary,smoothed, skeleton_uint8,smoothed_skel,edges,line_image):
    plt.figure(figsize=(26, 6))
    
    plt.subplot(1, 6, 1)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(smoothed, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(skeleton_uint8, cmap='gray')
    plt.title("Skeleton")
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(smoothed_skel, cmap='gray')
    plt.title("Smoothed Skeleton")
    plt.axis('off')
    
    plt.subplot(1, 6, 5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')
    
    plt.subplot(1, 6, 6)
    plt.imshow(line_image)
    plt.title("Detected Lines and Intersections")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
from skimage.morphology import skeletonize
    
def detect_intersections_and_lines_cv(inverted_image, reduced_image, eps=8, min_samples=1, min_line_length=30, max_line_gap=30, threshold_angle=25, showplot=True):
    print('starting computer vision apporach')
    
    # Apply Gaussian blur to smooth the edges
    kernel_size = 5
    smoothed = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 2)

    # Apply skeletonization
    skeleton = skeletonize(smoothed > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Apply Gaussian blur to smooth the edges
    kernel_size = 5
    smoothed_skel = cv2.GaussianBlur(skeleton_uint8, (kernel_size, kernel_size), 2)


    # Edge and line detection and intersection detections for lines that intersect at e.g. an angle > 20°
    # Apply Canny Edge Detection
    low_threshold = 10
    high_threshold = 200
    edges = cv2.Canny(smoothed_skel, low_threshold, high_threshold)

    # Hough Transform parameters
    rho = 1.5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20 # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 30  # minimum number of pixels making up a line
    #max_line_gap = 30  # maximum gap in pixels between connectable line segments
    line_image = np.copy(smoothed) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)



    # Detect intersections and calculate angles
    intersections = []
    if lines is not None:
        num_lines = len(lines)
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                line1 = lines[i][0]
                line2 = lines[j][0]
                intersect = line_intersection(line1, line2)
                if intersect:
                    angle = calculate_angle(line1, line2)
                    if angle > 25:
                        intersections.append(intersect)

    # Perform clustering and get the cluster centers
    cluster_centers_cv, labels = cluster_nodes(intersections, eps=eps, min_samples=min_samples)


    # Reinitialize line_image to draw combined intersections
    line_image = np.zeros_like(reduced_image)

    # Draw the detected lines on the line_image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the combined intersections on the line image
    for point in cluster_centers_cv:
        cv2.circle(line_image, tuple(map(int, point)), 5, (0, 255, 0), -1)

    print("Intersections detected and plotted.")

    # Plot the results
    line_detection_plot(inverted_image, smoothed, skeleton_uint8,smoothed_skel,edges,line_image)
    plot_cluster_centers(reduced_image, cluster_centers_cv, label='internal nodes cv')

    
    # Return detected lines and clustered intersection centers
    return np.array(cluster_centers_cv), lines
    
#%% principle stresses
import matplotlib.patches as mpatches

# Function to plot principal stresses at element centers
def plot_principal_stresses(element_list, x, scale=0.1):
    plt.figure()
    ax = plt.gca()

    for i, e in enumerate(element_list):
        sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
        sigma_1_vector = sigma_1 * np.array([np.cos(alpha), np.sin(alpha)])
        sigma_2_vector = sigma_2 * np.array([-np.sin(alpha), np.cos(alpha)])
        center = e.element_center()

        if x[i] > 0.5:
            ax.quiver(center[0], center[1], sigma_1_vector[0], sigma_1_vector[1], 
                      color='r', angles='xy', scale_units='xy', scale=scale, 
                      width=0.001, headwidth=4, headaxislength=4)
            ax.quiver(center[0], center[1], sigma_2_vector[0], sigma_2_vector[1], 
                      color='b', angles='xy', scale_units='xy', scale=scale, 
                      width=0.001, headwidth=4, headaxislength=4)
    
    # ax.set_xlim(0, 20)  # Limit x-axis range
    # ax.set_ylim(0, 15)  # Limit y-axis range
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Principal Stresses at Element Centers')
    plt.legend(["Sigma_1", "Sigma_2"], loc="best")
    plt.grid(True)
    plt.show()


# Function to plot tension and compression zones
def plot_tension_compression_zones(element_list, x):
    plt.figure()
    ax = plt.gca()

    sigma_t = []
    sigma_c = []
    tension_patch = mpatches.Patch(color='red', label='Tension')
    compression_patch = mpatches.Patch(color='blue', label='Compression')

    for i, e in enumerate(element_list):
        sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()

        if x[i] > 0.5:
            if abs(sigma_1) > abs(sigma_2):
                sigma_t.append(sigma_1)
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='red', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)
            else:
                sigma_c.append(sigma_2)
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='blue', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)

    plt.legend(handles=[tension_patch, compression_patch], loc="best")
    plt.title('Tension and Compression Zone')
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

    return np.mean(sigma_t) if sigma_t else 0, np.mean(sigma_c) if sigma_c else 0


# Function to plot Fang2023 criteria for nodal zones
def plot_nodal_zones_fang(element_list, x, sigma_t_avg, sigma_c_avg):
    plt.figure()
    ax = plt.gca()
    tension_patch = mpatches.Patch(color='red', label='Tension')
    compression_patch = mpatches.Patch(color='blue', label='Compression')
    nodal_patch = mpatches.Patch(color='green', label='Nodal Zone')

    for i, e in enumerate(element_list):
        sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
        total_sigma = sigma_1 + sigma_2

        if x[i] > 0.5:
            if total_sigma > 0.8 * sigma_t_avg and total_sigma < 1.2 * sigma_t_avg:
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='red', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)
            elif total_sigma > 1.2 * sigma_c_avg and total_sigma < 0.8 * sigma_c_avg:
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='blue', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)
            else:
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='green', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)

    plt.legend(handles=[tension_patch, compression_patch, nodal_patch], loc="best")
    plt.title('Nodal Zone Detection (Fang2023)')
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


# Function to plot nodal zones based on alternative criteria
def plot_nodal_zones_alternative(element_list, x):
    plt.figure()
    ax = plt.gca()

    for i, e in enumerate(element_list):
        sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()

        if x[i] > 0.5:
            if 0.25 * abs(sigma_2) < abs(sigma_1) < 4 * abs(sigma_2):
                coords = [n.coords for n in e.nodes]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax.fill(xs, ys, color='red', zorder=5)
                ax.plot(xs, ys, color="black", zorder=6, linewidth=0.5)

    # ax.set_xlim(0, 4)  # Limit x-axis range
    # ax.set_ylim(-1, 1)  # Limit y-axis range
    plt.title('Nodal Zone Detection (Alternative Criteria)')
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()
    
    
import plotly.graph_objects as go

# Function to create an interactive 3D scatter plot of principal stress angles
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

def plot_principal_stress_angles(element_list, x):
    # Set Plotly renderer to open in the browser
    pio.renderers.default = 'browser'  # Ensure it opens in the browser

    # Arrays to store the element centers (x, y) and the principal stress angles (alpha)
    x_vals = []
    y_vals = []
    alpha_vals = []

    # Function to convert radians to degrees
    def normalize_angle(alpha):
        degrees = np.degrees(alpha)  # Convert radians to degrees
        return degrees

    # Collect the data
    for i, e in enumerate(element_list):
        sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
        center = e.element_center()

        if x[i] > 0.5:  # Apply the same condition for filtering elements
            # Store the center coordinates (x, y) and the normalized principal stress angle
            x_vals.append(center[0])
            y_vals.append(center[1])
            alpha_vals.append(normalize_angle(alpha))  # Convert to degrees

    # Create the interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals, 
        y=y_vals, 
        z=alpha_vals, 
        mode='markers',
        marker=dict(
            size=5,             # Marker size
            color=alpha_vals,   # Color based on the angle in degrees
            colorscale='Viridis',  # Colormap
            opacity=0.8         # Opacity of the points
        )
    )])

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Principal Stress Angle (degrees)',  # Updated to reflect degrees
        ),
        title='Principal Stress Angles (Degrees) at Element Centers',
        autosize=False,
        width=800,
        height=800,
    )

    # Show the interactive plot in the browser
    fig.show()


#%% clustering based on principal stress states and coordinates
from sklearn.preprocessing import StandardScaler

# Function to prepare data (extract principal stresses, angles, and center coordinates)
def prepare_and_normalize_element_data(element_list, x, alpha_threshold=1.3, x_filter=0.5):
    """
    Prepare data for principal stresses, angles, and center coordinates of elements.
    Normalize the data for clustering.

    Parameters:
    - element_list: List of elements to process.
    - x: List of x-coordinates (or other relevant data) used as a filter for elements.
    - alpha_threshold: Threshold for adjusting the angle `alpha`. Default is 1.3 radians.
    - x_filter: Filter for x-coordinate values. Only elements where x > x_filter are considered.

    Returns:
    - elements: A NumPy array containing [sigma_1, sigma_2, alpha, center_x, center_y] for each element.
    - elements_scaled: Scaled version of the `elements` array (for clustering).
    """
    elements = []
    
    for i, e in enumerate(element_list):
        if x[i] > x_filter:  # Only process elements where x[i] > x_filter
            sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
            center = e.element_center()

            # Adjust alpha if necessary
            if alpha > alpha_threshold:
                alpha -= np.pi  # Adjust the angle alpha if it exceeds the threshold

            # Append the data [sigma_1, sigma_2, alpha, center_x, center_y] to elements
            elements.append([sigma_1, sigma_2, alpha, center[0], center[1]])

    # Convert elements to a NumPy array
    elements = np.array(elements)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    elements_scaled = scaler.fit_transform(elements)

    return elements, elements_scaled

# Function to perform DBSCAN clustering and plot the results
def cluster_and_plot(element_list, x, eps=0.22, min_samples=20):
    """
    Apply DBSCAN clustering on scaled data and plot the results using original coordinates.

    Parameters:
    - elements: Original data (with principal stresses, alpha, x, and y).
    - elements_scaled: Scaled version of the original data for clustering.
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    This function plots the clustering results.
    """
    
    # Step 1: Prepare and normalize the element data
    elements, elements_scaled = prepare_and_normalize_element_data(element_list, x)

    # Step 2: Perform clustering and plot
    
    # Apply DBSCAN clustering on the scaled data
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(elements_scaled)

    # Extract cluster labels
    labels = db.labels_

    # Plot the clusters in the original space (using original x and y)
    plt.figure()
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise points
            col = [0, 0, 0, 1]

        # Select the points that belong to this cluster
        class_member_mask = (labels == k)
        xy = elements[class_member_mask]  # Use the original elements (sigma_1, sigma_2, alpha, x, y)

        # Plot the cluster in original space (x and y center coordinates)
        plt.scatter(xy[:, 3], xy[:, 4], c=[tuple(col)], label=f'Cluster {k}' if k != -1 else 'Noise', s=50)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'DBSCAN Clustering in Original Space (eps={eps})')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.grid(True)
    plt.show()
#%% nodes from BCs

# Helper function to check if three points are collinear (on the same line)
def are_collinear(p1, p2, p3):
    """Returns True if the points are collinear."""
    return np.isclose((p2[1] - p1[1]) * (p3[0] - p1[0]), (p3[1] - p1[1]) * (p2[0] - p1[0]))

# Function to find and merge collinear lines, and return remaining nodes as Node objects
def find_and_merge_collinear_lines(nodes, node_list):
    """
    Find and merge collinear lines into start and end points, return single Node objects.
    
    Parameters:
    - nodes: List of coordinates.
    - node_list: List of original Node objects from which these coordinates are taken.

    Returns:
    - lines: Merged collinear lines (start and end points).
    - single_nodes: List of remaining Node objects that are not collinear.
    """
    if len(nodes) < 2:
        # If less than 2 nodes, return no lines, but return the original Node objects
        return [], [node_list[i] for i, coords in enumerate(nodes)]
    
    lines = []
    single_nodes = set(tuple(n) for n in nodes)  # Store all nodes initially as single tuples

    # Sort nodes by x and y to simplify processing
    nodes = sorted(nodes, key=lambda p: (p[0], p[1]))
    
    # Create a flag array to mark visited nodes
    visited = [False] * len(nodes)

    # Iterate through each point and form groups of collinear points
    for i in range(len(nodes)):
        if visited[i]:
            continue
        
        ref_point = tuple(nodes[i])  # Convert to tuple
        collinear_group = [ref_point]
        visited[i] = True
        
        # Check for collinear points with the reference point
        for j in range(i + 1, len(nodes)):
            if visited[j]:
                continue
            
            for k in range(j + 1, len(nodes)):
                if are_collinear(ref_point, nodes[j], nodes[k]):
                    collinear_group.append(tuple(nodes[j]))  # Convert to tuple
                    collinear_group.append(tuple(nodes[k]))  # Convert to tuple
                    visited[j] = True
                    visited[k] = True
        
        # Sort the collinear group by x or y and find the start and end points
        if len(collinear_group) > 1:
            collinear_group = list(set(collinear_group))  # Remove duplicates
            collinear_group.sort(key=lambda p: (p[0], p[1]))  # Sort by x and y
            
            # Add the start and end points of the line
            start_point = collinear_group[0]
            end_point = collinear_group[-1]
            lines.append((start_point, end_point))
            
            # Remove the points that form part of the lines from the single nodes set
            for point in collinear_group:
                if point in single_nodes:
                    single_nodes.remove(point)
    
    # Convert remaining single node coordinates to Node objects
    single_node_objs = [node_list[i] for i, coords in enumerate(nodes) if tuple(coords) in single_nodes]
    
    return lines, single_node_objs


# Main function to find nodes with loads or fixed supports and process them
def process_supports_and_loads(s):
    """
    Find all nodes that have a load or a fixed support, merge collinear points, and return Node objects.
    Supports and loads are combined into one list of nodes.
    """

    # Step 1: Find and process support nodes (nodes that are fixed)
    support_nodes = [n for n in s.nodes if any(n.fixed)]  # Filter only fixed nodes
    support_coords = [n.coords for n in support_nodes]  # Get coordinates of support nodes
    support_lines, single_support_nodes = find_and_merge_collinear_lines(support_coords, support_nodes)  # Process collinear supports
    
    # Step 2: Find and process load nodes (nodes that have non-zero forces)
    load_nodes = [n for n in s.nodes if np.any(n.forces != 0)]  # Filter only load nodes
    load_coords = [n.coords for n in load_nodes]  # Get coordinates of load nodes
    load_lines, single_load_nodes = find_and_merge_collinear_lines(load_coords, load_nodes)  # Process collinear loads
    
    # Step 3: Combine single nodes (support and load) into a single list
    all_nodes = single_support_nodes + single_load_nodes  # Combine the lists
    
    # Step 4: Return the results in real-world coordinates
    return support_lines, load_lines, all_nodes  


def transform_and_plot_bcs(nodes, support_lines, load_lines, transformation_realworld_to_image, dimensions, dimensions_img, topopt_result):
    """
    Transforms boundary conditions (nodes and lines) from real-world coordinates to image space, 
    and plots the transformed supports and loads on top of the topology optimization result.

    Parameters:
    - nodes: List of Node objects (combined support and load nodes).
    - support_lines: List of tuples representing collinear support lines in real-world coordinates.
    - load_lines: List of tuples representing collinear load lines in real-world coordinates.
    - transformation_realworld_to_image: Function to transform real-world coordinates to image coordinates.
    - dimensions: Real-world dimensions.
    - dimensions_img: Image dimensions.
    - topopt_result: The topology optimization result image (for plotting the boundary conditions on top).
    """

    # Step 1: Transform all node coordinates and classify them into support or load nodes
    support_points_img = []
    load_points_img = []

    for node in nodes:
        coords_img = transformation_realworld_to_image(node.coords, dimensions, dimensions_img)
        if any(node.fixed):  # If the node is a support
            support_points_img.append(coords_img)
        elif np.any(node.forces != 0):  # If the node has non-zero forces (load)
            load_points_img.append(coords_img)
    
    # Step 2: Transform the support and load lines to image coordinates
    support_lines_img = []
    for line in support_lines:
        line_img = [transformation_realworld_to_image(point, dimensions, dimensions_img) for point in line]
        support_lines_img.append(line_img)

    load_lines_img = []
    for line in load_lines:
        line_img = [transformation_realworld_to_image(point, dimensions, dimensions_img) for point in line]
        load_lines_img.append(line_img)
    
    # Step 3: Plot the transformed points and lines on top of the topology optimization result
    plt.figure()
    plt.imshow(topopt_result)  # Show the topology optimization result as the background

    # Plot the transformed support points (as red circles)
    support_points_img = np.array(support_points_img)
    if len(support_points_img) > 0:
        plt.scatter(support_points_img[:, 0], support_points_img[:, 1], c='red', label="Support Points", marker='x', s=100)

    # Plot the transformed load points (as blue squares)
    load_points_img = np.array(load_points_img)
    if len(load_points_img) > 0:
        plt.scatter(load_points_img[:, 0], load_points_img[:, 1], c='green', label="Load Points", marker='x', s=100)

    # Plot the transformed support lines (as red lines)
    for line in support_lines_img:
        line = np.array(line)
        plt.plot(line[:, 0], line[:, 1], color='red', linewidth=2, label="Support Lines")

    # Plot the transformed load lines (as blue lines)
    for line in load_lines_img:
        line = np.array(line)
        plt.plot(line[:, 0], line[:, 1], color='green', linewidth=2, label="Load Lines")
    
    # Set plot settings
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.title("Boundary Conditions on TopOpt Result")
    plt.show()
    
    return support_points_img, support_lines_img, load_points_img, load_lines_img

#%% nodes on line support

def nodes_on_line_support(image, support_lines_img):
    """
    Walks along each support line and scans the pixels adjacent to it. If black pixels are detected 
    to the left or right of the current pixel, the current pixel is added to a black interval. 
    Allows for small gaps between black pixels to merge intervals.
    
    Parameters:
    - image: The image to scan.
    - support_lines_img: List of support lines (in image coordinates) where each line consists of start and end points.
    - max_gap: Maximum gap (in pixels) allowed between black pixel intervals to still consider them part of the same interval.
    
    Returns:
    - black_intervals: A list of intervals (each interval being a list of merged consecutive pixels) for each support line.
    - centers_of_intervals: A list of the centers of the black intervals for each support line.
    """
    # Ensure the image is in grayscale format
    if len(image.shape) == 3:  # If the image has 3 channels (e.g., RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
    node_candidates = []

    # Iterate over each support line in image coordinates
    for line in support_lines_img:
        start_point, end_point = line
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Use Bresenham's algorithm to get the pixels along the line between start and end points
        pixels_on_line = list(bresenham(int(x1), int(y1), int(x2), int(y2)))
        
        # Scan each pixel along the line
        for i, (px, py) in enumerate(pixels_on_line):
            if is_mostly_black(image, px, py, radius=5,lower_threshold=0.2,upper_threshold=1):
                #add to node candidates
                node_candidates.append((px, py))

    return node_candidates


def bresenham(x1, y1, x2, y2):
    """
    Bresenham's Line Algorithm to return a list of pixel coordinates between two points (x1, y1) and (x2, y2).
    """
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        pixels.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels
