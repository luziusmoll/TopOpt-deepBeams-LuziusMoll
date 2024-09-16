from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import cv2


def reduce_image_colors(image, grayscale_threshold=102, disp_bc=True):
    """
    Reduces an image to four colors: white, black, red, and green.

    Parameters:
    - image: The input image in RGB format (256, 256, 3).
    - grayscale_threshold: Threshold for converting grayscale to black or white.
                           Values below 40% (102 in [0, 255]) become black.
    - disp_bc: Boolean flag. If True, keep red and green pixels; otherwise, set them to white.

    Returns:
    - reduced_image: The image reduced to the four colors.
    """

    # Convert the image to grayscale to apply the black and white threshold
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize the result image as white
    reduced_image = np.ones_like(image) * 255  # Start with a white image

    # Apply threshold to determine black pixels
    black_mask = grayscale_image < grayscale_threshold
    reduced_image[black_mask] = [0, 0, 0]  # Set to black

    # Identify red and green pixels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Red color mask
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Green color mask
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    if disp_bc:
        # Apply red and green masks to the result image
        reduced_image[red_mask > 0] = [255, 0, 0]  # Set red areas
        reduced_image[green_mask > 0] = [0, 255, 0]  # Set green areas
    else:
        # Set red and green areas to white
        reduced_image[red_mask > 0] = [255, 255, 255]  # Make red areas white
        reduced_image[green_mask > 0] = [255, 255, 255]  # Make green areas white

    return reduced_image


# def is_black(image, x, y, window_size=2, threshold=0.8):
#     """
#     Checks if a pixel at (x, y) is black by considering its surrounding pixels.

#     Parameters:
#     - image: The input grayscale image.
#     - x, y: The coordinates of the pixel to check.
#     - window_size: The size of the neighborhood window (e.g., 5x5 or larger).
#     - threshold: The percentage of surrounding pixels that need to be black to classify the center as black.

#     Returns:
#     - True if the pixel and its surroundings are mostly black, False otherwise.
#     """
#     half_window = window_size // 2
    
#     # Extract the surrounding window
#     x_start = max(0, x - half_window)
#     x_end = min(image.shape[1], x + half_window + 1)
#     y_start = max(0, y - half_window)
#     y_end = min(image.shape[0], y + half_window + 1)
    
#     window = image[y_start:y_end, x_start:x_end]
    
#     # Count the number of black pixels in the window
#     black_pixels = np.sum(window == 0)
    
#     # Calculate the ratio of black pixels to the total number of pixels
#     total_pixels = window.size
#     black_ratio = black_pixels / total_pixels
    
#     # If more than 'threshold' percentage of pixels are black, classify as black
#     return black_ratio >= threshold


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


def plot_cluster_centers(image, cluster_centers):
    """
    Plots the cluster centers on the image.

    Parameters:
    - image: The input grayscale image.
    - cluster_centers: A list of (x, y) tuples representing the cluster centers.
    """
    plt.imshow(image, cmap='gray')

    # Plot all cluster centers as red points
    for (x, y) in cluster_centers:
        plt.scatter(x, y, color='red', s=5)

    plt.axis('off')
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
    img = convert_to_binary(img)  # Ensure the image is binary
    prev_img = np.zeros_like(img)
    iteration = 0
    while not np.array_equal(img, prev_img):
        prev_img = img.copy()
        # subiteration 0
        img = thinning_iteration(img, 0)
        plt.imshow(img, cmap='gray')
        plt.title(f"Thinning Iteration {iteration} subiteration 0")
        plt.show()
        # subiteration 1
        img = thinning_iteration(img, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Thinning Iteration {iteration} subiteration 1")
        plt.show()
        iteration += 1
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

#%% line detection

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