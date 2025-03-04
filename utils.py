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


def convert_to_binary(img):
    """Converts a grayscale or RGB image to binary (0 and 255) in uint8 format."""
    # Convert RGB to grayscale if the image has multiple channels
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)
    
    # Convert grayscale to binary (0 or 255)
    binary_img = np.where(img > 128, 255, 0).astype(np.uint8)
    
    return binary_img


def invert_image(img):
    """Inverts a binary image (0 and 255)."""
    return 255 - img


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



def preprocess_image(s, path, target_size, grayscale_threshold=102):
    """
    Reduces an image to four colors: white, black, red, and green.
    Also extracts the coordinates of blue dots used for image boundary marking.

    Parameters:
    - image: The input image in RGB format (256, 256, 3).
    - grayscale_threshold: Threshold for converting grayscale to black or white.
                           Values below 40% (102 in [0, 255]) become black.
    - disp_bc: Boolean flag. If True, keep red and green pixels; otherwise, set them to white.

    Returns:
    - reduced_image: The image reduced to the four colors.
    - blue_dot_coordinates: A tuple with the coordinates of the two blue dots (bottom-left and top-right).
    """
    
    plot_variable, dimensions = s.plot4(deformed=False, disp_bc=False, disp_corner=True)


    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{s.name}.png"
    filepath = os.path.join(path, filename)


    plot_variable.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plot_variable.savefig(filepath, bbox_inches='tight', pad_inches=0)

    print(f"Saved plot as {filepath}")


    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_rescaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)


    # Pad the image with white borders
    color = [255, 255, 255]  # White color
    image_padded = cv2.copyMakeBorder(image_rescaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    image_normalized = image_padded / 255.0


    # Convert back to uint8 format to save the image
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    image = image_uint8


    # Convert the image to grayscale to apply the black and white threshold
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to smooth the grayscale image
    blurred_image = cv2.GaussianBlur(grayscale_image, (3, 3), 20)
    
    # Apply a fixed threshold to the blurred image
    _, binary_image = cv2.threshold(blurred_image, grayscale_threshold, 255, cv2.THRESH_BINARY)



    # Find the yellow dots in the image and their coordinates using a flexible HSV range
    # Yellow color in HSV has hue around 60° (range of 20-40°)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow in HSV
    upper_yellow = np.array([40, 255, 255])  # Upper bound for yellow in HSV
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Get the coordinates of yellow pixels
    yellow_coords = np.column_stack(np.where(yellow_mask > 0))
    
    if len(yellow_coords) >= 2:
        # Sort by y descending (highest y first), then by x ascending (smallest x first)
        sorted_yellow_coords = sorted(yellow_coords, key=lambda x: (-x[0], x[1]))  # Sort by y descending, then by x ascending
        # Bottom-left is the first point (largest y, smallest x), and top-right is the last point (smallest y, largest x)
        bottom_left = sorted_yellow_coords[0]
        top_right = sorted_yellow_coords[-1]
        yellow_dot_coordinates = ([bottom_left[1], bottom_left[0]], [top_right[1], top_right[0]])
    else:
        # If yellow dots are not found (unexpected), return None
        yellow_dot_coordinates = None
    
    return binary_image, dimensions, yellow_dot_coordinates


def transformation_realworld_to_image(coords, dimensions, dimensions_img):
    """
    Transforms real-world coordinates to image pixel coordinates.
    
    Parameters:
    - coords: Real-world coordinates [x, y] to be transformed.
    - dimensions: [[min_xs, min_ys], [max_xs, max_ys]] in real-world coordinates.
    - dimensions_img: ((bottom_left_x, bottom_left_y), (top_right_x, top_right_y)) in image pixel coordinates.

    Returns:
    - Image pixel coordinates [x_img, y_img].
    """
    # Extract real-world dimensions
    (min_xs, min_ys), (max_xs, max_ys) = dimensions
    #print('dimensions', dimensions)
    
    # Extract image dimensions (in pixel coordinates)
    (bottom_left_x_img, bottom_left_y_img), (top_right_x_img, top_right_y_img) = dimensions_img
    #print('dimensions_img', dimensions_img)
    
    # Real-world to image scaling factors
    scale_x = (top_right_x_img - bottom_left_x_img) / (max_xs - min_xs)
    scale_y = (top_right_y_img - bottom_left_y_img) / (max_ys - min_ys)

    # Apply the scaling transformation
    x_img = bottom_left_x_img + (coords[0] - min_xs) * scale_x
    y_img = bottom_left_y_img + (coords[1] - min_ys) * scale_y

    return [x_img, y_img]
   
    
def transformation_image_to_realworld(coords, dimensions, dimensions_img):
    """
    Transforms image pixel coordinates to real-world coordinates.
    
    Parameters:
    - coords: Image pixel coordinates [x_img, y_img] to be transformed.
    - dimensions: [[min_xs, min_ys], [max_xs, max_ys]] in real-world coordinates.
    - dimensions_img: ((bottom_left_x, bottom_left_y), (top_right_x, top_right_y)) in image pixel coordinates.

    Returns:
    - Real-world coordinates [x_real, y_real].
    """
    # Extract real-world dimensions
    (min_xs, min_ys), (max_xs, max_ys) = dimensions
    
    # Extract image dimensions (in pixel coordinates)
    (bottom_left_x_img, bottom_left_y_img), (top_right_x_img, top_right_y_img) = dimensions_img

    # Image to real-world scaling factors
    scale_x = (max_xs - min_xs) / (top_right_x_img - bottom_left_x_img)
    scale_y = (max_ys - min_ys) / (top_right_y_img - bottom_left_y_img)

    # Apply the reverse scaling transformation
    x_real = min_xs + (coords[0] - bottom_left_x_img) * scale_x
    y_real = min_ys + (coords[1] - bottom_left_y_img) * scale_y

    return [x_real, y_real]    


#%% clustering of node candidates

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
        plt.scatter(x, y, color='blue', label=label, marker='o', s=10)

    # Ensure only unique entries in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removes duplicate labels by using a dictionary

    # Set plot settings
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.title("Extracted Nodes")
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


#%% circular node detection filter


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


def is_mostly_black(image, center_x, center_y, radius, lower_threshold=0.8, upper_threshold=1):
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


def black_ratio(image, center_x, center_y, radius):
    """
    Checks if the percentage of black pixels within a circle around (center_x, center_y)
    is between the specified lower and upper thresholds.

    Parameters:
    - image: The input grayscale image.
    - center_x, center_y: The coordinates of the center of the circle.
    - radius: The radius of the circle.

    Returns:
    - The ratio of black pixels within the circle.
    """
    black_pixel_count = 0
    total_pixel_count = 0

    # Iterate over a square of size 2*radius around the center
    for x in range(int(center_x - radius), int(center_x + radius + 1)):
        for y in range(int(center_y - radius), int(center_y + radius + 1)):
            # Check if the pixel is within the circle of the specified radius
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= radius:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
                    total_pixel_count += 1
                    if image[y, x] == 0:  # Black pixel
                        black_pixel_count += 1

    if total_pixel_count == 0:  # Avoid division by zero
        return 0.0

    black_ratio = black_pixel_count / total_pixel_count

    return black_ratio


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

    # filter out small black segments
    filtered_segments = []
    for i in range(len(segments)):
        curr_start, curr_end = segments[i]
        if curr_end - curr_start > min_angle_diff:
            filtered_segments.append((curr_start, curr_end))

    # Merge segments with small small with in between
    merged_segments = [filtered_segments[0]]  # Start with the first segment

    for i in range(1, len(filtered_segments)):
        prev_start, prev_end = merged_segments[-1]
        curr_start, curr_end = filtered_segments[i]

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


def check_cone_for_white_pixels(image, center, radius, start_angle, end_angle, white_threshold=0.05):
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

    # Iterate over a square of size 2*radius around the center
    for x in range(x_center - radius, x_center + radius + 1):
        for y in range(y_center - radius, y_center + radius + 1):
            # Check if the pixel is within the circle of the specified radius
            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            if distance <= radius:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
                    # Check if the pixel is within the angle range
                    angle = np.arctan2(y - y_center, x - x_center)
                    if angle < 0:
                        angle += 2 * np.pi
                    if start_angle <= angle <= end_angle:
                        total_pixels += 1
                        if image[y, x] == 255:  # White pixel
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


def find_node_candidates(image, radius=5, min_angle_diff=np.deg2rad(25), white_threshold=0.05):
    """
    Finds node candidates based on a fixed radius and neighborhood check, with additional filtering for white pixels in cones.

    Parameters:
    - image: The input grayscale image.
    - radius: The minimum radius for the circle
    - white_threshold: The percentage threshold of white pixels to filter segments.

    Returns:
    - node_candidates: A list of (x, y) tuples representing the node positions.
    - segments_info: A dictionary containing the start and end angles for each segment for each node.
    """
    node_candidates = []
    segments_info = {}
    radii = []
    
    # time taking
    time_filter_seg = 0
    time_circ_check = 0
    time_radius = 0
    time_merge_segs = 0

    # Ensure the image is in grayscale format
    if len(image.shape) == 3:  # If the image has 3 channels (e.g., RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    rows, cols = image.shape  # Ensure this is 2D
    min_radius = radius
    for x in range(min_radius, cols - min_radius):
        if x%20==0:
            print('column',x, 'from', image.shape[0])
        for y in range(min_radius, rows - min_radius):
            # print('line',y)
            if is_black(image, x, y): # check that current pixel is black
                radius = min_radius
                ratio = black_ratio(image, x, y, radius)
                #print(ratio)
                if ratio > 0.85:
                    # print(ratio)
                    start_time_radius = time.time()
                    while ratio>0.85 and radius < 100: # find the smallest circle around the current pixel that has approx 85% black pixels
                        radius +=1
                        ratio = black_ratio(image, x, y, radius)
                        
                    end_time_radius = time.time()
                    time_radius += end_time_radius-start_time_radius
                    
                    start_circ_check = time.time()
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
                    
                    end_circ_check = time.time()
                    time_circ_check += end_circ_check-start_circ_check
                    
                    
                    # Merge segments with small angle differences
                    start_merge_seg_check = time.time()
                    merged_segments = merge_segments(segments, min_angle_diff)
                    end_merge_seg_check = time.time()
                    time_merge_segs += end_merge_seg_check-start_merge_seg_check
    
                    # Filter segments by checking the cone for white pixels
                    start_time = time.time()
                    filtered_segments = filter_segments_by_cone(image, (x, y), radius, merged_segments, white_threshold)
                    end_time = time.time()
                    time_filter_seg += end_time-start_time
                    
                    # filtered_segments=merged_segments
                    
                    # Classify the center as a node based on the number of filtered segments
                    if classify_node_by_segments(filtered_segments):
                        node_candidates.append((x, y))
                        segments_info[(x, y)] = filtered_segments
                        radii.append(radius)
    
    print('find radius time:', time_radius)   
    print('circumference check for segments total time:', time_circ_check) 
    print('merge segments time:', time_merge_segs)              
    print('filter segments total time:', time_filter_seg)
    
    
    print(len(node_candidates), 'node candidates found')
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


def plot_node_with_segments(image, node, radius, segments,i=None):
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
    plt.title(f"node candidate {i}")
    plt.show()

    
#%% Xia 2020a

def thinning_iteration(img, iter_num, stm):
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

    # dont remove pixels at boundary conditions:
    for n in stm.node_list:
        j,i = int(n.coords_img[0]), int(n.coords_img[1])
        if marker[i,j]==1:
            marker[i,j] = 0

    img[marker == 1] = 0  # Remove the marked pixels
    
    return img * 255  # Convert back to 0 and 255


def zhang_suen_thinning(img, stm):
    print('thinning started')
    img = convert_to_binary(img)  # Ensure the image is binary
    prev_img = np.zeros_like(img)
    iteration = 0
    while not np.array_equal(img, prev_img):
        prev_img = img.copy()
        # subiteration 0
        img = thinning_iteration(img, 0, stm)
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Thinning Iteration {iteration} subiteration 0")
        # plt.show()
        # subiteration 1
        img = thinning_iteration(img, 1, stm)
        # plt.imshow(img, cmap='gray')
        # plt.title(f"Thinning Iteration {iteration} subiteration 1")
        # plt.show()
        iteration += 1
        
    print('thinning finished after ', iteration,' iterations')
    return img


# Function to check if a 3x3 block contains a pattern (pattern must be contained within the block)
def contains_pattern(block, pattern, match):
    #exact match
    if match == 'exact':
        return np.array_equal(block, pattern)
    #containing the pattern is sufficient
    else:
        return np.all((pattern == 0) | (block == pattern))

# Function to check if a 3x3 block matches any pattern (including rotations)
def matches_pattern(block, match):
    # Define the node patterns as binary 3x3 matrices
    patterns = [
        np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]]),  # Pattern 1 
        np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 2
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),  # Pattern 3
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 4
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])   # Pattern 5
    ]
    for pattern in patterns:
        # Check all four rotations (0, 90, 180, 270 degrees)
        for _ in range(4):
            if contains_pattern(block, pattern, match):
                return True
            # Rotate pattern 90 degrees
            pattern = np.rot90(pattern)
    return False

# Sliding window over the skeletonized image
def detect_nodes(skeletonized_image, match=None):
    rows, cols = skeletonized_image.shape
    node_positions = []

    # Slide a 3x3 window over the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            block = skeletonized_image[i-1:i+2, j-1:j+2]  # Extract 3x3 block
            if matches_pattern(block, match):
                node_positions.append((j, i))  # Add node position if pattern matches

    return node_positions


#%% path following

#follow_skeleton_path_bfs(node, skeleton_img, visited, stm)
def follow_skeleton_path_bfs(start_node, skeleton_img, visited, stm):
    """
    Follows the skeleton path starting from a node using BFS and stops 
    following a path once another node is detected, but continues exploring other paths.
    Adds the connection to the adjacency list of the stm object.
    """
    cx, cy = map(int, start_node.coords_skel)  # Start node as integers
    
    # Mark the current node as visited
    visited[cx, cy] = -2
    
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
            # new_end_nodes = []
            
            # If the neighbor is another node and not the starting node 
            if visited[nx, ny] >= 0 and visited[nx, ny] != start_node.id:  # (nx, ny) != (int(start_node.coords_skel[0]), int(start_node.coords_skel[1]))
                connection = (start_node.id, visited[nx, ny])  # new connection
             
                if connection in stm.adjacency_list or (connection[1], connection[0]) in stm.adjacency_list or start_node.id ==  visited[nx, ny]:
                    pass
                else:
                    stm.adjacency_list.append(connection)
                
                # # Debug visualization of detected connections
                # debug_img[ny, nx] = 127  # Mark path in gray
                # debug_img_uint8 = (debug_img * 255).astype(np.uint8)
                # debug_img_colored = cv2.cvtColor(debug_img_uint8, cv2.COLOR_GRAY2RGB)
                # cx_int, cy_int = tuple(map(int, (cx, cy)))
                # nx_int, ny_int = tuple(map(int, (nx, ny)))
                # cv2.circle(debug_img_colored, (nx_int, ny_int), 3, (0, 255, 0), -1)  # Mark endpoint in green
                # cv2.line(debug_img_colored, (int(start_node.coords_skel[0]), int(start_node.coords_skel[1])), (nx_int, ny_int), (255, 0, 0), 2)
                
                # # Display the debug image (intermediate steps)
                # plt.imshow(debug_img_colored)
                # # plt.title(f" Connection from ({int(start_node[0])}, {int(start_node[1])}) to ({nx_int}, {ny_int})")
                # plt.axis('off')
                # plt.show()
                
                # new_end_nodes.append([nx, ny])
                
                # Remove all pixels in a 3x3 grid around the found node from the queue to make sure the path is no longer followed
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        grid_x, grid_y = nx + dx, ny + dy
                        if (grid_x, grid_y) in queue:
                            queue.remove((grid_x, grid_y))
                
                break
            

            # If the neighbor is part of the skeleton and unvisited
            elif visited[nx, ny] == -1:
                visited[nx, ny] = -2  # Mark as visited
                queue.append((nx, ny))  # Add to queue to continue BFS search
                
                # # Debug visualization of intermediate steps
                # debug_img[ny, nx] = 127  # Mark path in gray
                # debug_img_uint8 = (debug_img * 255).astype(np.uint8)
                # debug_img_colored = cv2.cvtColor(debug_img_uint8, cv2.COLOR_GRAY2RGB)
                # cx_int, cy_int = tuple(map(int, (cx, cy)))
                # nx_int, ny_int = tuple(map(int, (nx, ny)))
                # cv2.circle(debug_img_colored, (nx_int, ny_int), 3, (0, 255, 0), -1)  # Mark current pixel
                # cv2.line(debug_img_colored, (int(start_node.coords_skel[0]), int(start_node.coords_skel[1])), (nx_int, ny_int), (255, 0, 0), 2)
                
                # plt.imshow(debug_img_colored)
                # # plt.title(f"Following path from ({int(start_node[0])}, {int(start_node[1])}) to ({nx_int}, {ny_int})")
                # plt.axis('off')
                # plt.show()
                


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


def generate_truss_structure_bfs(stm, skeleton_img):
    """
    Generates a truss-like structure by detecting straight-line connections between nodes along the skeleton.
    This version uses BFS to explore all possible paths and visualize the path-following process.
    
    Parameters:
    - nodes: List of node coordinates
    - skeleton_img: Binary skeletonized image (1 for skeleton pixels, 0 for background)
    
    Returns:
    - 
    """

    # search connection for each node
    for node in stm.node_list:
        
        # Initialize visited matrix for each node
        visited = -np.ones(skeleton_img.shape, dtype=int)
        
        # Mark all node locations in the visited matrix with their id (special value for nodes)
        for n in stm.node_list:
            # Round the node coordinates to the nearest integer
            nx, ny = round(n.coords_skel[0]), round(n.coords_skel[1])
            visited[nx, ny] = n.id  # Mark this node as a special value
        
        # Find connections starting from the current node using BFS
        follow_skeleton_path_bfs(node, skeleton_img, visited, stm)
        # truss_connections.extend(connections)
    

#%% computer vision(not presented in the thesis due to bad results)

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
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

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
    plt.legend(["Sigma 1", "Sigma 2"], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(False)
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
    
    start_time_princ = time.time()
    for i, e in enumerate(element_list):
        if x[i] > x_filter:  # Only process elements where x[i] > x_filter
            
            sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
            
            
            center = e.element_center()

            # Adjust alpha if necessary
            if alpha > alpha_threshold:
                alpha -= np.pi  # Adjust the angle alpha if it exceeds the threshold

            # Append the data [sigma_1, sigma_2, alpha, center_x, center_y] to elements
            elements.append([sigma_1, sigma_2, alpha, center[0], center[1]])
            
    end_time_princ = time.time()
    time_princ_recov =end_time_princ-start_time_princ
    print('time_princ_recov', time_princ_recov)
    # Convert elements to a NumPy array
    elements = np.array(elements)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    elements_scaled = scaler.fit_transform(elements)

    return elements, elements_scaled


def cluster_and_plot(element_list, x, min_cluster_size=10):
    """
    Apply HDBSCAN clustering on scaled data and plot the results using original coordinates.

    Parameters:
    - elements: Original data (with principal stresses, alpha, x, and y).
    - elements_scaled: Scaled version of the original data for clustering.
    - min_cluster_size: The minimum size of clusters to be considered valid.

    This function plots the clustering results.
    """
    
    # Step 1: Prepare and normalize the element data
    start_time_feature = time.time()
    elements, elements_scaled = prepare_and_normalize_element_data(element_list, x)
    end_time_feature = time.time()
    time_feature = end_time_feature-start_time_feature
    print('time_feature', time_feature)
    # Step 2: Perform HDBSCAN clustering and plot
    
    start_clustering = time.time()
    # Apply HDBSCAN clustering on the scaled data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(elements_scaled)  # Get the cluster labels
    end_clustering = time.time()
    cluster_time = end_clustering-start_clustering
    print('clustering time ', cluster_time)
    
    start_plot = time.time()
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
        plt.scatter(xy[:, 3], xy[:, 4], c=[tuple(col)], label=f'Cluster {k}' if k != -1 else 'Noise', s=10)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'HDBSCAN Clustering in Original Space (min_cluster_size={min_cluster_size})')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.grid(True)
    plt.show()
    
    end_plot = time.time()
    plot_time = end_plot-start_plot
    print('plot time' ,plot_time)
    


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


#%% shape optimization


def domain_penalty1(system, design_boundary, holes, node_weight=1.0, beam_weight=1.0, penalty_scale=1.0, ax=None, plot=None):
    """
    Calculate total domain penalty for a system, considering node and beam penalties, with visualization.

    Parameters:
    - system: The structural system containing nodes and elements (beams).
    - design_boundary: Polygon defining the design boundary.
    - holes: List of Polygon objects defining holes.
    - node_weight: Weight for node penalties.
    - beam_weight: Weight for beam penalties.
    - penalty_scale: Scaling factor for penalties.
    - ax: Matplotlib axis object for visualization. If None, no plot is generated.

    Returns:
    - Total domain penalty (weighted sum of node and beam penalties).
    """
    total_node_penalty = 0
    total_beam_penalty = 0

    if ax is not None:
        # Plot the design boundary
        x, y = design_boundary.exterior.xy
        ax.plot(x, y, 'blue', label='Design Boundary')

        # Plot holes
        if holes is not None:
            for hole in holes:
                x, y = hole.exterior.xy
                ax.plot(x, y, 'black', linestyle='--', label='Hole')

    # Node penalties
    for node in system.nodes:
        point = Point(node.coords)
        penalty = 0

        # Check if the node is outside the design boundary
        if not design_boundary.covers(point):
            nearest_point = design_boundary.exterior.interpolate(design_boundary.exterior.project(point))
            distance = design_boundary.exterior.distance(point)
            penalty += penalty_scale * distance

            # Visualization
            if ax is not None:
                ax.plot(
                    [node.coords[0], nearest_point.x],
                    [node.coords[1], nearest_point.y],
                    'orange', label='Node Penalty (Boundary)' if 'Node Penalty (Boundary)' not in ax.get_legend_handles_labels()[1] else ""
                )
                ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        # Check if the node is inside a hole
        if holes is not None:
            for hole in holes:
                if hole.contains(point):
                    nearest_point = hole.exterior.interpolate(hole.exterior.project(point))
                    distance = hole.exterior.distance(point)
                    penalty += penalty_scale * distance
    
                    # Visualization
                    if ax is not None:
                        ax.plot(
                            [node.coords[0], nearest_point.x],
                            [node.coords[1], nearest_point.y],
                            'orange', label='Node Penalty (Hole)' if 'Node Penalty (Hole)' not in ax.get_legend_handles_labels()[1] else ""
                        )
                        ax.scatter(node.coords[0], node.coords[1], color='orange', label='Invalid Node' if 'Invalid Node' not in ax.get_legend_handles_labels()[1] else "")

        total_node_penalty += penalty

    # Beam penalties
    for beam in system.elements:
        beam_segment = LineString([beam.nodes[0].coords, beam.nodes[1].coords])
        penalty = 0
        valid_segments = [beam_segment]  # Start with the full beam as valid

        if holes is not None:
            for hole in holes:
                if beam_segment.intersects(hole):
                    intersection = beam_segment.intersection(hole.exterior)
    
                    points = [Point(beam.nodes[0].coords)]
                    if intersection.geom_type == 'Point':
                        points.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        points.extend(intersection.geoms)  # Extract points from MultiPoint
                    points.append(Point(beam.nodes[1].coords))
    
                    points = sorted(points, key=lambda p: beam_segment.project(p))
                    subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]
    
                    valid_segments = []
                    for segment in subsegments:
                        if segment.within(hole):
                            length_invalid = segment.length
                            penalty += penalty_scale * length_invalid
                            if ax is not None:
                                ax.plot(*segment.xy, color='red', linewidth=2, label='Invalid Beam (Hole)' if 'Invalid Beam (Hole)' not in ax.get_legend_handles_labels()[1] else "")
                        else:
                            valid_segments.append(segment)

        for segment in valid_segments:
            if not design_boundary.covers(segment):
                intersection = segment.intersection(design_boundary.exterior)

                points = [Point(segment.coords[0])]
                if intersection.geom_type == 'Point':
                    points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    # points.extend(list(intersection))
                    points.extend(intersection.geoms) 
                points.append(Point(segment.coords[-1]))

                points = sorted(points, key=lambda p: segment.project(p))
                subsegments = [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]

                valid_segments = []
                for subsegment in subsegments:
                    if not design_boundary.covers(subsegment):
                        length_invalid = subsegment.length
                        penalty += penalty_scale * length_invalid
                        if ax is not None:
                            ax.plot(*subsegment.xy, color='red', linewidth=2, label='Invalid Beam (Boundary)' if 'Invalid Beam (Boundary)' not in ax.get_legend_handles_labels()[1] else "")
                    else:
                        valid_segments.append(subsegment)

        for segment in valid_segments:
            if ax is not None:
                ax.plot(*segment.xy, color='green', linewidth=2, label='Valid Beam' if 'Valid Beam' not in ax.get_legend_handles_labels()[1] else "")

        total_beam_penalty += penalty

    # Weighted total penalty
    total_penalty = node_weight * total_node_penalty + beam_weight * total_beam_penalty

    if ax is not None and total_penalty>0 or plot:
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("Domain Penalty Visualization")
        ax.legend()
        ax.grid(True)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()
    else:
         plt.close()

    return total_penalty


def shape_optimization(max_iter, system_shape_opt, design_boundary, holes, l_B=1, penalty_nodes=1, penalty_ele=1, output_file=None, l_min=None):

    # Get bounding box from design boundary
    min_x, min_y, max_x, max_y = design_boundary.bounds
    
    # Compute dimensions
    dimension = [max_x - min_x, max_y - min_y]  # [width, height]
    
    # step size
    eta = min(dimension)/200
    
    if l_min is None:
        l_min = eta*10

    # Optimization parameters
    #max_iter = 200  # Maximum number of iterations
    tolerance = 1e-10  # Convergence tolerance for objective
    move_limit = eta*100  # Maximum allowable change in design update
  
    iteration = 0
    obj_prev = float('inf')  # Initialize previous objective to a large value
    objective_hist = [] 
    strain_energy_N_hist = []
    strain_energy_B_hist = []
    domain_penalty_hist = []

    while iteration < max_iter:
        
        # step 0: solve FE
        system_shape_opt.solve_FE()
        
        # Step 1: Compute objective and sensitivity
        fig, ax = plt.subplots(figsize=(10, 8))
        
        total_penalty = domain_penalty1(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0, ax=ax)
        
        u_N, u_B = system_shape_opt.strain_energy_beam_truss()
        strain_E =  u_N + u_B
        objective = u_N + l_B*u_B + total_penalty
        
        
        objective_hist.append(objective)
        strain_energy_N_hist.append(u_N)
        strain_energy_B_hist.append(u_B)
        domain_penalty_hist.append(total_penalty)

        d_c = np.zeros(len(system_shape_opt.nodes) * 2)  # Sensitivity array for x and y coordinates

        for i, node in enumerate(system_shape_opt.nodes):
            # Skip fixed nodes (coordinates are fixed)
            if any(node.fixed):
                d_c[i * 2] = 0
                d_c[i * 2 + 1] = 0
                # print(f"Node {i} fixed")
                continue

            # Skip nodes with non-zero external forces (coordinates are fixed)
            if np.linalg.norm(node.forces) > 0:  # Check if forces are non-zero
                d_c[i * 2] = 0
                d_c[i * 2 + 1] = 0
                # print(f"Node {i} has external forces")
                continue

            # Compute sensitivity for internal nodes
            for coord_index in range(2):  # x and y coordinates
                original_value = node.coords[coord_index]
                node.coords[coord_index] += eta  # Perturb the coordinate
                system_shape_opt.solve_FE()  # Recalculate system with perturbed geometry
                total_penalty = domain_penalty1(system_shape_opt, design_boundary, holes, node_weight=penalty_nodes, beam_weight=penalty_ele, penalty_scale=1.0)

                
                u_N, u_B = system_shape_opt.strain_energy_beam_truss()
                obj_var = u_N + l_B*u_B + total_penalty
                d_c[i * 2 + coord_index] = (obj_var - objective) / eta
                node.coords[coord_index] = original_value  # Reset to original

        # Step 2: Update nodal coordinates in the negative d_c direction
        # Update nodal coordinates with capped step size
        for i, node in enumerate(system_shape_opt.nodes):
            if any(node.fixed) or np.linalg.norm(node.forces) > 0:
                continue  # Skip fixed or loaded nodes
        
            # Compute step size for x and y directions
            step_x = max(min(eta * 10* d_c[i * 2], move_limit), -move_limit)  # Cap step size by move_limit_x
            step_y = max(min(eta * 10* d_c[i * 2 + 1], move_limit), -move_limit)  # Cap step size by move_limit_y
        
            # Update node coordinates with bounds checks
            node.coords[0] = max(min(node.coords[0] - step_x, max_x), min_x)
            node.coords[1] = max(min(node.coords[1] - step_y, max_y), min_y)


        # system_shape_opt.plot_deformed_stm_sf(100,10)

        # Step 3: Check convergence
        change = abs(obj_prev - objective)

        if change < tolerance:
            print("Convergence achieved!")
            break
        

        # Step 4: Check for merged nodes
        system_shape_opt.delete_short_elements(l_min)
       
        # Optional: Plot the deformed structure at each iteration
        if (iteration) % 10 ==0:
            print(f"Iteration {iteration + 1}")
            
            print(f"number of dofs {system_shape_opt.nr_dofs}")
            system_shape_opt.plot_deformed_stm_sf(100, scale=10, title=f'Iteration: {iteration}')
            print(f"Strain energy: {strain_E}, Change: {change}")
        
        
        # Update previous objective and iteration counter
        obj_prev = objective
        iteration += 1

    # Final output
    print("Optimization completed.")
    print(f"Final strain energy: {strain_E}")
    # calculate ratio of normal forces
    sts = system_shape_opt.sts()
    # formatted_sts = [f"{value[0]:.4f}" for value in sts]
    # print("STS per Element:", ", ".join(formatted_sts))
    print('sts:', np.mean(sts))

    # data
    iterations = np.arange(1, len(objective_hist) + 1)
    strain_energy_N =  np.array(strain_energy_N_hist)
    strain_energy_B =  np.array(strain_energy_B_hist)
    strain_energy = strain_energy_N + strain_energy_B
       
    
    # Plot 1: Strain Energies
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, strain_energy, label='Strain Energy', color='blue', linewidth=1.5)
    plt.plot(iterations, strain_energy_N, label='Axial Strain Energy', color='green', linestyle='--', linewidth=1.5)
    plt.plot(iterations, strain_energy_B, label='Bending Strain Energy', color='red', linestyle=':', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Strain Energy')
    plt.title('Strain Energy Histories')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("strain_energy_histories.pdf", format="pdf", dpi=300)
    plt.show()
    
    
    # Plot 2: Objective, strain energy, and Domain Penalty
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, objective_hist, label='Objective', color='purple', linewidth=1.5)
    plt.plot(iterations, strain_energy, label='Strain Energy', color='blue', linestyle='--', linewidth=1.5)
    plt.plot(iterations, domain_penalty_hist, label='Domain Penalty', color='orange', linestyle=':', linewidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Objective, Strain Energy, and Domain Penalty')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("objective_compliance_penalty_histories.pdf", format="pdf", dpi=300)
    plt.show()

