import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.morphology import skeletonize
from skimage.util import invert
from matplotlib import gridspec

def combined_plot(s, obj_hist, x):
    fig = plt.figure(figsize=(18, 5))  # Overall figure size
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Adjust the middle plot width if needed
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Plotting the optimized structure using plot3 method
    s.plot3(ax=ax1, deformed=False)
    ax1.set_title('Mesh Plot')
    ax1.set_aspect('equal')  # Set to 'equal' to maintain original scale (otherwise 'auto')
    
    # Plotting Objective History
    ax2.plot(obj_hist)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective')
    ax2.set_title('Objective History')
    ax2.grid(True)
    
    # Plotting the distribution of x
    ax3.hist(x, bins=30, alpha=0.75)
    ax3.set_title('Histogram of x')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

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

def save_plot_as_image(plot_variable, folder_name="topopt_ressults"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    existing_files = [f for f in os.listdir(folder_name) if f.endswith('.png')]
    if existing_files:
        numbers = [int(f.split('.')[0].split('_')[-1]) for f in existing_files]
        highest_number = max(numbers)
    else:
        highest_number = 0

    new_number = highest_number + 1
    filename = f"topology_plot_{new_number}.png"
    filepath = os.path.join(folder_name, filename)

    plot_variable.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plot_variable.savefig(filepath, bbox_inches='tight', pad_inches=0)

    print(f"Saved plot as {filepath}")
    return filepath

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape

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

    transformation_rule = {
        'scale': scale,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'original_shape': original_shape,
        'target_size': target_size
    }

    # Convert back to uint8 format to save the image
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    return image_uint8, transformation_rule

def save_preprocessed_image(image, folder_name="preprocessed_images"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    existing_files = [f for f in os.listdir(folder_name) if f.endswith('.png')]
    if existing_files:
        numbers = [int(f.split('.')[0].split('_')[-1]) for f in existing_files]
        highest_number = max(numbers)
    else:
        highest_number = 0

    new_number = highest_number + 1
    filename = f"preprocessed_plot_{new_number}.png"
    filepath = os.path.join(folder_name, filename)

    # Save the image
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved preprocessed image as {filepath}")

    return filepath

def reduce_image_colors(image, grayscale_threshold=102, disp_bc=True):
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

    # -----------------------------------------------
    # Find the blue dots in the image and their coordinates using a flexible HSV range
    # Blue color in HSV has hue around 120° (range of 100-140°)
    lower_blue = np.array([100, 100, 100])  # Lower bound for blue in HSV
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue in HSV
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Get the coordinates of blue pixels
    blue_coords = np.column_stack(np.where(blue_mask > 0))

    if len(blue_coords) >= 2:
        # Sort by y descending (highest y first), then by x ascending (smallest x first)
        sorted_blue_coords = sorted(blue_coords, key=lambda x: (-x[0], x[1]))  # Sort by y descending, then by x ascending
        # Bottom-left is the first point (largest y, smallest x), and top-right is the last point (smallest y, largest x)
        bottom_left = sorted_blue_coords[0]
        top_right = sorted_blue_coords[-1]
        blue_dot_coordinates = ([bottom_left[1], bottom_left[0]], [top_right[1], top_right[0]])
    else:
        # If blue dots are not found (unexpected), return None
        blue_dot_coordinates = None
    

    return reduced_image, blue_dot_coordinates

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
    print('dimensions', dimensions)
    
    # Extract image dimensions (in pixel coordinates)
    (bottom_left_x_img, bottom_left_y_img), (top_right_x_img, top_right_y_img) = dimensions_img
    print('dimensions_img', dimensions_img)
    
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
   

#%% old    

# def apply_transformation(coord, transformation_rule):
#     scale = transformation_rule['scale']
#     top = transformation_rule['top']
#     left = transformation_rule['left']
#     original_shape = transformation_rule['original_shape']

#     x, y = coord
#     x_new = int(x * scale) + left
#     y_new = int(y * scale) + top

#     return (x_new, y_new)

# def reverse_transformation(coord, transformation_rule):
#     scale = transformation_rule['scale']
#     top = transformation_rule['top']
#     left = transformation_rule['left']

#     x, y = coord
#     x_orig = (x - left) / scale
#     y_orig = (y - top) / scale

#     return (int(x_orig), int(y_orig))

# # real world coordinates
# def real_world_dimension(node_list):
#     min_x = min(node.coords[0] for node in node_list)
#     max_x = max(node.coords[0] for node in node_list)
#     min_y = min(node.coords[1] for node in node_list)
#     max_y = max(node.coords[1] for node in node_list)

#     return [min_x, max_x, min_y, max_y]

# # Function to convert pixel coordinates to real-world coordinates
# def pixel_to_real_world(dimensions, coord, scale_x, scale_y, padding_top, padding_left):
#     x_pixel, y_pixel = coord
#     x_real = dimensions[0] + (-padding_left + x_pixel) * scale_x
#     y_real = dimensions[3] - (-padding_top + y_pixel) * scale_y
#     return x_real, y_real

def invert_image(img):
    """Inverts a binary image (0 and 255)."""
    return 255 - img

def convert_to_binary(img):
    """Converts a grayscale or RGB image to binary (0 and 255) in uint8 format."""
    # Convert RGB to grayscale if the image has multiple channels
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)
    
    # Convert grayscale to binary (0 or 255)
    binary_img = np.where(img > 128, 255, 0).astype(np.uint8)
    
    return binary_img


