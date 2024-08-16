import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from system import System
from mesh import Mesh

# parameters:
volfrac=0.4
penalty = 3
x_min = 1e-3
ft=0        # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
max_iteration = 30 
mesh_ind_filter = True
regular_mesh = False

# r_min needs to be according to  dimensions of the problem
if regular_mesh == True:
    r_min = 4
else:
    r_min = 0.25
    
    
# set up geometry as defined in mesh_test
node_list, element_list  = Mesh.create(regular_mesh)
print('number of elements:', len(element_list))

for e in element_list:
    e.E = 30000
    e.nu = 0.15

# volume fraction for all elements is set to volfrac
x = np.ones(len(element_list),dtype=float)*volfrac

# Set up FE problem
s = System(node_list, element_list, x, penalty, x_min)


# Apply boundary conditions to structure
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
#s.fix_node_by_coord([0,-1])
#s.fix_node_by_coord([4,-1])
if regular_mesh == True:
    s.load_point([80,20],[0,-0.1])
else:
    s.load_point([4,-1],[0,-1])
    #s.load_point([1.5,-1],[0,-1])
#s.load_line(np.array([60,0.0]), np.array([60,3.0]),forces=np.array([0.0,-0.01]))
s.apply_dirichlet_bc()



#%% Convolution operator for mesh independency filtering
""" from sigmund2001: A 99 line topology optimization code written in Matlab: eq6"""

# distance between current element and all others
element_centers = s.element_centers()
element_centers = np.array(element_centers)

dist = []
for i in range(len(element_list)):
    dist_ij = []
    for j in range(len(element_list)):
        dist_x = element_centers[i,0]-element_centers[j,0]
        dist_y = element_centers[i,1]-element_centers[j,1]
        dist_ij.append(np.sqrt(dist_x**2 + dist_y**2))
    dist.append(dist_ij)

    
# convolution operator H_f
H_f = r_min * np.ones([len(x),len(x)]) - dist
# set negativ values (elements outside of r_min) to zero
H_f[H_f < 0] = 0

        
       

#%% Optimality criterion
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(n_ele,x,volfrac,dc,dv,g):
    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(n_ele)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # possibility to define passive areas
        if regular_mesh == True: 
            for ely in range(40):
                for elx in range(80):
                    if np.sqrt((ely-20)**2 + (elx-30)**2) < 10:
                        xnew[elx*40+ely] = x_min
        
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (xnew,gt)


#%% Actual optimization 
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

# Set loop counter and gradient vectors 
loop=0
obj_hist = []
change=1
dv = np.ones(len(element_list))
dc = np.ones(len(element_list))
ce = np.ones(len(element_list))
# The following must be initialized to use the NGuyen/Paulino OC approachgls
xold=x.copy()
xPhys=x.copy()
g=0 
while change>0.001 and loop<max_iteration: 
    loop=loop+1
    
    # Solve FE problem
    print(loop)
    u = s.solve_FE() 
    
    #K_g = s.K_global()
    #print(K_g)
    # Objective and sensitivity
    obj=s.compliance()
    obj_hist.append(obj)
    # according to sigmund2001 eq4 (no filter)
    dc=s.sensitivity_compliance()  
    
    # according to sigmund2001 eq5 (with filter)
    if mesh_ind_filter == True:
        dc_filtered = []
        for i in range(len(element_list)):
            dc_filtered_i = 1 / x[i] * np.sum(H_f[:,i]) * np.sum( H_f[:,i] * x * dc)
            dc_filtered.append(dc_filtered_i)
            
        dc= dc_filtered
        
    
    dv = np.ones(len(element_list))
    # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
    # if ft==0:
    #     dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
    # elif ft==1:
    #     dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
    #     dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
    # Optimality criteria
    xold[:]=x
    (x[:],g)=oc(len(element_list),x,volfrac,dc,dv,g)
    # pass new x vector to system
    s.x = x
    # Filter design variables
    # if ft==0:   xPhys[:]=x
    # elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
    # Compute the change by the inf. norm 
    change=np.linalg.norm(x.reshape(len(element_list),1)-xold.reshape(len(element_list),1),np.inf)
    # Write iteration history to screen (req. Python 2.6 or newer)
    print('obj:',obj)
    print('change:', change)
    print('mean x:',np.mean(x))
    #print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    if (loop - 1) % 3 == 0:
        #s.plot2(deformed=False)
        s.plot2(deformed=True)
 

s.plot2(deformed=False)

#%% combined plot for obsidian

def combined_plot(s):
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
    
combined_plot(s)


#%% processing and saving of image
import cv2
import os
from skimage.morphology import skeletonize
from skimage.util import invert

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


# Generate and save the plot
plot_variable = s.plot4(deformed=False) 
image_path = save_plot_as_image(plot_variable)

# Preprocess the image
target_size = 256  # Example target size

preprocessed_image, transformation_rule = preprocess_image(image_path, target_size)

plot_image(preprocessed_image)
#%% 
import cv2
import numpy as np

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

# Example usage:

reduced_image = reduce_image_colors(preprocessed_image, grayscale_threshold=102, disp_bc=False)
plot_image(reduced_image)



# boundary conditions should be passed on from the input in a usable format
load_points = ([4,-1],[],)
line_loads = ()
fixed_points = ()
fixed_lines = ([[0,-1],[0,1]],[])


import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_black(image, x, y, window_size=2, threshold=0.8):
    """
    Checks if a pixel at (x, y) is black by considering its surrounding pixels.

    Parameters:
    - image: The input grayscale image.
    - x, y: The coordinates of the pixel to check.
    - window_size: The size of the neighborhood window (e.g., 5x5 or larger).
    - threshold: The percentage of surrounding pixels that need to be black to classify the center as black.

    Returns:
    - True if the pixel and its surroundings are mostly black, False otherwise.
    """
    half_window = window_size // 2
    
    # Extract the surrounding window
    x_start = max(0, x - half_window)
    x_end = min(image.shape[1], x + half_window + 1)
    y_start = max(0, y - half_window)
    y_end = min(image.shape[0], y + half_window + 1)
    
    window = image[y_start:y_end, x_start:x_end]
    
    # Count the number of black pixels in the window
    black_pixels = np.sum(window == 0)
    
    # Calculate the ratio of black pixels to the total number of pixels
    total_pixels = window.size
    black_ratio = black_pixels / total_pixels
    
    # If more than 'threshold' percentage of pixels are black, classify as black
    return black_ratio >= threshold

def merge_segments(segments, min_angle_diff=np.deg2rad(15)):
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

    # for i in range(1, len(segments)):
    #     prev_start, prev_end = merged_segments[-1]
    #     curr_start, curr_end = segments[i]

    #     # If the angle difference between the previous segment's end and the current segment's start is small, merge them
    #     if curr_start - prev_end < min_angle_diff:
    #         # Merge by extending the previous segment's end to the current segment's end
    #         merged_segments[-1] = (prev_start, curr_end)
    #     elif curr_start - prev_end - 2*np.pi < min_angle_diff and curr_start - prev_end - 2*np.pi > 0:
    #         merged_segments[-1] = (prev_start, curr_end)
    #     else:
    #         # Otherwise, add the current segment as a new segment
    #         merged_segments.append((curr_start, curr_end))
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

def find_node_candidates(image, radius=10):
    """
    Finds node candidates based on a fixed radius and neighborhood check.

    Parameters:
    - image: The input grayscale image.
    - radius: The fixed radius for the circle.

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
                merged_segments = merge_segments(segments)

                # Classify the center as a node based on the number of segments
                if classify_node_by_segments(merged_segments):
                    node_candidates.append((x, y))
                    segments_info[(x, y)] = merged_segments

    return node_candidates, segments_info

def classify_node_by_segments(segments):
    """
    Classifies a center point as a node based on the number of segments detected.
    """
    # Check if there are more than 2 segments
    if len(segments) > 2:
        return True
    
    # If there are 2 or fewer segments, it's not a node
    return False

import numpy as np
import matplotlib.pyplot as plt

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
            plt.plot(x_circ, y_circ, 'ro', markersize=2)  # Mark segment points as red

    plt.scatter(x, y, color='green', s=50)  # Mark the node center as green
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
        plt.scatter(x, y, color='green', s=5)

    plt.axis('off')
    plt.show()
    
    

radius = 15

# Run the node detection function
node_candidates, segments_info = find_node_candidates(reduced_image, radius=radius)

# Select one node candidate to visualize
if node_candidates:
    i = 20  # Example index
    selected_node = node_candidates[i]
    segments = segments_info[selected_node]
    
    # Plot the circle and detected segments for the selected node
    plot_node_with_segments(reduced_image, selected_node, radius=radius, segments=segments)
    
    print(segments_info[selected_node])
else:
    print("No node candidates detected.")
    


# After detecting node candidates, call the function to plot them
plot_all_nodes(reduced_image, node_candidates)


#%%

def plot_all_nodes(image, node_candidates):
    """
    Plots all detected node candidates on the image.
    
    Parameters:
    - image: The input grayscale image.
    - node_candidates: A list of (x, y) tuples representing the node positions.
    """
    plt.imshow(image, cmap='gray')

    # Plot all nodes as green points
    for (x, y) in node_candidates:
        plt.scatter(x, y, color='green', s=20)

    plt.axis('off')
    plt.show()

# After detecting node candidates, call the function to plot them
plot_all_nodes(reduced_image, node_candidates)


# Select one node candidate to visualize
if node_candidates:
    for i in range(len(node_candidates)):
        selected_node = node_candidates[i]
        segments = segments_info[selected_node]
        
        # Plot the circle and detected segments for the selected node
        plot_node_with_segments(reduced_image, selected_node, radius=radius, segments=segments)
        
    print(segments_info[selected_node])
else:
    print("No node candidates detected.")

#%%

import matplotlib.pyplot as plt

def plot_image(image):
    """
    Plots an image with the shape (256, 256, 3).

    Parameters:
    - image: The image array to be plotted. It should be in the format (256, 256, 3).
    """
    plt.imshow(image)
    plt.axis('off')  # Hide axis labels and ticks
    plt.show()

# Draw the detected lines on the line_image
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

# Draw the combined intersections on the line image
for point in combined_intersections:
    cv2.circle(line_image, point, 5, (0, 255, 0), -1)
# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(smoothed, cmap='gray')
plt.title("Grayscale Image of Skeleton")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(line_image)
plt.title("Detected Lines")
plt.axis('off')

plt.tight_layout()
plt.show() 


print("Intersections detected and plotted.")



#%%

# Function to convert pixel coordinates to real-world coordinates
def pixel_to_real_world(coord, scale_x, scale_y, padding_top, padding_left):
    x_pixel, y_pixel = coord
    x_real = dimensions[0] + (-padding_left + x_pixel) * scale_x
    y_real = dimensions[3] - (-padding_top + y_pixel) * scale_y
    return x_real, y_real

# Convert combined intersections back to the original image space
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in combined_intersections]
print(f"Intersection: {original_intersection_coordinates}")

# Find dimensions of the structure in the original image
original_image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# Identify non-white pixels (gray level less than 255)
non_white_pixels = np.where(gray_image < 255)

# Calculate the real-world dimensions and the original image dimensions
dimensions = real_world_dimension(node_list)
real_world_width = dimensions[1] - dimensions[0]
real_world_height = dimensions[3] - dimensions[2]
original_width = transformation_rule['original_shape'][1]
original_height = transformation_rule['original_shape'][0]

# Find the extreme coordinates
top_most = np.min(non_white_pixels[0])
bottom_most = np.max(non_white_pixels[0])
left_most = np.min(non_white_pixels[1])
right_most = np.max(non_white_pixels[1])

# Calculate padding
padding_top = top_most
padding_bottom = gray_image.shape[0] - bottom_most
padding_left = left_most
padding_right = gray_image.shape[1] - right_most
 
scale_x = real_world_width / (original_width-padding_left-padding_right)
scale_y = real_world_height / (original_height-padding_bottom-padding_top)

# Convert the original space coordinates to real-world coordinates using the scale factors
real_world_coordinates = [pixel_to_real_world(coord, scale_x, scale_y, padding_top, padding_left) for coord in original_intersection_coordinates]

# Print the real-world coordinates
for idx, coord in enumerate(real_world_coordinates):
    print(f"Intersection {idx+1}: {coord}")



#%% plotting nodes on original image


# Load the original image
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Convert the combined intersections back to the original image space
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in combined_intersections]

# Plot the original image
plt.figure(figsize=(10, 10))
plt.imshow(original_image)
plt.title("Original Image with Intersection Points")

# Overlay the intersection coordinates
for idx, coord in enumerate(original_intersection_coordinates):
    plt.scatter(*coord, color='blue', s=60)  # s is the size of the marker
    plt.text(coord[0], coord[1], str(idx + 1), color='blue', fontsize=20)

plt.axis('off')
plt.show()

# Optionally save the plotted image with intersections
output_image_path_with_intersections = "original_image_with_intersections.png"


#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def ellipse_mask(image_shape, center, axes, angle):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    return mask

def is_node_in_ellipse(node, center, axes, angle):
    # Transform node coordinates to the ellipse's coordinate system
    x, y = node
    cx, cy = center
    a, b = axes
    theta = np.radians(angle)
    
    # Rotate node to align with ellipse axes
    x_rot = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    y_rot = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
    
    # Check if the point is inside the ellipse
    if (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1:
        return True
    return False

def is_truss_between_nodes(image, node1, node2, nodes, threshold=0.8):
    center = ((node1[0] + node2[0]) // 2, (node1[1] + node2[1]) // 2)
    axes = (int(np.linalg.norm(np.array(node1) - np.array(center))), 10)  # semi-major and semi-minor axes
    angle = np.degrees(np.arctan2(node2[1] - node1[1], node2[0] - node1[0]))
    
    mask = ellipse_mask(image.shape, center, axes, angle)
    dark_pixel_count = np.sum(image[mask == 1] < 127)
    total_pixel_count = np.sum(mask == 1)
    
    # Check if any other node is inside the ellipse
    for node in nodes:
        if node != node1 and node != node2:
            if is_node_in_ellipse(node, center, axes, angle):
                return False

    return (dark_pixel_count / total_pixel_count) > threshold

# Load the original image
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Convert the combined intersections back to the original image space
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in combined_intersections]

# Convert original image to grayscale for truss detection
gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# Determine trusses between nodes
trusses = []
for i, node1 in enumerate(original_intersection_coordinates):
    for j, node2 in enumerate(original_intersection_coordinates):
        if i < j:
            if is_truss_between_nodes(gray_original_image, node1, node2, original_intersection_coordinates):
                trusses.append((i, j))

# Plot the original image with detected trusses
plt.figure(figsize=(10, 10))
plt.imshow(original_image)
plt.title("Original Image with Detected Trusses and Intersection Points")

# Overlay the intersection coordinates
for idx, coord in enumerate(original_intersection_coordinates):
    plt.scatter(*coord, color='blue', s=80)  # s is the size of the marker
    plt.text(coord[0], coord[1], str(idx + 1), color='blue', fontsize=30)

# Overlay trusses
for (i, j) in trusses:
    node1 = original_intersection_coordinates[i]
    node2 = original_intersection_coordinates[j]
    plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='yellow', linewidth=2)

plt.axis('off')
plt.show()

# Optionally save the plotted image with intersections and trusses
output_image_path_with_trusses = "original_image_with_trusses.png"

print(f"Image with intersections and trusses saved to {output_image_path_with_trusses}")

# Save node coordinates and truss elements to a CSV file
csv_filename = "trusses_and_nodes.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header for nodes
    writer.writerow(["Node Index", "X Coordinate", "Y Coordinate"])
    for idx, coord in enumerate(original_intersection_coordinates):
        writer.writerow([idx + 1, coord[0], coord[1]])
    
    # Write header for trusses
    writer.writerow([])
    writer.writerow(["Truss Index", "Start Node", "End Node"])
    for truss_idx, (start_node, end_node) in enumerate(trusses):
        writer.writerow([truss_idx + 1, start_node + 1, end_node + 1])

print(f"CSV file with nodes and trusses saved to {csv_filename}")

