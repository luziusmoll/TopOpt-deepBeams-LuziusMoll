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

def save_plot_as_image(plot_variable, folder_name="preprocessed_images"):
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

    # plt.figure(figsize=(6, 6))
    # plt.imshow(image_normalized)
    # plt.axis('off')
    # plt.title("Preprocessed Image")
    # plt.show()
    
    image_uint8 = (image_normalized * 255).astype(np.uint8)
    
    return image_uint8, transformation_rule

def apply_transformation(coord, transformation_rule):
    scale = transformation_rule['scale']
    top = transformation_rule['top']
    left = transformation_rule['left']
    original_shape = transformation_rule['original_shape']

    x, y = coord
    x_new = int(x * scale) + left
    y_new = int(y * scale) + top

    return (x_new, y_new)

def reverse_transformation(coord, transformation_rule):
    scale = transformation_rule['scale']
    top = transformation_rule['top']
    left = transformation_rule['left']

    x, y = coord
    x_orig = (x - left) / scale
    y_orig = (y - top) / scale

    return (int(x_orig), int(y_orig))

# real world coordinates
def real_world_dimension(node_list):
    min_x = min(node.coords[0] for node in node_list)
    max_x = max(node.coords[0] for node in node_list)
    min_y = min(node.coords[1] for node in node_list)
    max_y = max(node.coords[1] for node in node_list)

    return [min_x, max_x, min_y, max_y]

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