import numpy as np
import cv2
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
    r_min = 0.25 # mesh one

    
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

# Mesh one
# s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
# #s.fix_node_by_coord([0,-1])
# #s.fix_node_by_coord([4,-1])
# if regular_mesh == True:
#     s.load_point([80,20],[0,-0.1])
# else:
#     s.load_point([4,-1],[0,-1])
#     #s.load_point([1.5,-1],[0,-1])
# #s.load_line(np.array([60,0.0]), np.array([60,3.0]),forces=np.array([0.0,-0.01]))

# Corbel 
r_min = 3
s.fix_line(np.array([0.0,0.0]), np.array([50.0,0.0]))
s.fix_line(np.array([0.0,270.0]), np.array([50.0,270.0]))
s.load_point([95,170],[0,-1])

# # wall with openings
# r_min = 2
# s.fix_node_by_coord([5,0])
# s.fix_node_by_coord([117.5,0], fix = [False,True])
# s.load_point([37.5,75],[0,-1])
# s.load_point([85,75],[0,-1])


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
            
        # with out this float division by 0 can occour in the while loop criteria
        # additional line compared to sigmund 200 line implementation
        if l1 + l2 == 0:
            return (xnew,gt)
            
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
            # additional if criteria compared to sigmund
            if x[i] * np.sum(H_f[:,i]) > 0:
                dc_filtered_i = 1 / x[i] * np.sum(H_f[:,i]) * np.sum( H_f[:,i] * x * dc)
            else:
                dc_filtered_i = dc[i]
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

# combined plot for obsidian
from image_processing_utils import combined_plot
combined_plot(s, obj_hist, x)


#%% processing and saving of image
from image_processing_utils import save_plot_as_image, plot_image, preprocess_image, apply_transformation, reverse_transformation, real_world_dimension, convert_to_binary, invert_image

# Generate and save the plot
plot_variable = s.plot4(deformed=False) 
image_path = save_plot_as_image(plot_variable)

# Preprocess the image
target_size = 256  # Example target size

preprocessed_image, transformation_rule = preprocess_image(image_path, target_size)

plot_image(preprocessed_image)
#%% extraction with my own node detection filter


# import utilities
from exctraction_utils import reduce_image_colors, cluster_nodes, plot_cluster_centers, find_node_candidates, plot_node_with_segments, plot_all_nodes


# boundary conditions should be passed on from the input in a usable format
# load_points = ([4,-1],[],)
# line_loads = ()
# fixed_points = ()
# fixed_lines = ([[0,-1],[0,1]],[])

threshold = 102
# reduce image colors to black, white and red, green if disp_bc is True
reduced_image = reduce_image_colors(preprocessed_image, grayscale_threshold=threshold, disp_bc=False)
plot_image(reduced_image)


# set filter radius
radius = 15 
min_angle_diff=20
# Run the node detection function

all_node_candidates = []
all_segments_info = {}
all_radii = []
for radius in range(2, 11):
    print('searching for candidates with radius', radius)
    node_candidates, segments_info, radii = find_node_candidates(reduced_image, radius=radius, min_angle_diff=np.deg2rad(min_angle_diff), white_threshold=0.05)
    
    # Extend the node candidates list instead of appending
    all_node_candidates.extend(node_candidates)
    
    # Merge segments_info dictionaries
    all_segments_info.update(segments_info)
    
    # Extend radii
    all_radii.extend(radii)

# After detecting node candidates, call the function to plot them
plot_all_nodes(reduced_image, all_node_candidates)


#%% clustering

# Set DBSCAN parameters
eps = 5  # Maximum distance for points to be considered in the same cluster
min_samples = 5  # Minimum number of points required to form a cluster

# Perform clustering and get the cluster centers
cluster_centers, labels = cluster_nodes(all_node_candidates, eps=eps, min_samples=min_samples)

# Plot the cluster centers on the image
plot_cluster_centers(reduced_image, cluster_centers)

# Optionally, print the cluster centers
print("Cluster Centers:", cluster_centers)

nodes = cluster_centers
#%% visualizing segments

# Select  or all node candidates to visualize the segments 
if all_node_candidates: 
    print('displaying a few node candidates together with the node detection filter')
    #for i in range(len(node_candidates)):
    for i in range(0,len(all_node_candidates),int(len(all_node_candidates)/10)):
        selected_node = all_node_candidates[i]
        segments = all_segments_info[selected_node]
        
        # Plot the circle and detected segments for the selected node
        plot_node_with_segments(reduced_image, selected_node, radius=all_radii[i], segments=segments)
        
else:
    print("No node candidates detected.")
        

#%% Skeletonization from zhang1984

threshold = 102
# reduce image colors to black, white and red, green if disp_bc is True
reduced_image = reduce_image_colors(preprocessed_image, grayscale_threshold=threshold, disp_bc=False)
plot_image(reduced_image)

from image_processing_utils import zhang_suen_thinning

# Convert the resized image from BGR (OpenCV default) to RGB for correct color display in Matplotlib
image_rgb = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)
binary_img = convert_to_binary(image_rgb)

# Invert the binary image to thin the black areas
inverted_img = invert_image(binary_img)

# Display the inverted image (optional)
plt.imshow(inverted_img, cmap='gray')
plt.axis('off')  # Optional: turn off axis
plt.show()

# Apply the Zhang-Suen thinning algorithm on the inverted image
thinned_img_inverted = zhang_suen_thinning(inverted_img)

# Invert the thinned image back to the original format
thinned_img = invert_image(thinned_img_inverted)

# Display the final thinned image
plt.imshow(thinned_img, cmap='gray')
plt.axis('off')  # Optional: turn off axis
plt.show()


#%% node detection patterns from Xia2020a

# Define the node patterns as binary 3x3 matrices
patterns = [
    np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]]),  # Pattern 1 (rotational equivalents not shown)
    np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 2
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),  # Pattern 3
    np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),  # Pattern 4
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])   # Pattern 5
]

# Function to check if a 3x3 block contains a pattern (pattern must be contained within the block)
def contains_pattern(block, pattern):
    return np.all((pattern == 0) | (block == pattern))

# Function to check if a 3x3 block matches any pattern (including rotations)
def matches_pattern(block, patterns):
    for pattern in patterns:
        # Check all four rotations (0, 90, 180, 270 degrees)
        for _ in range(4):
            if contains_pattern(block, pattern):
                return True
            # Rotate pattern 90 degrees
            pattern = np.rot90(pattern)
    return False

# Sliding window over the skeletonized image
def detect_nodes(skeletonized_image, patterns):
    rows, cols = skeletonized_image.shape
    node_positions = []

    # Slide a 3x3 window over the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            block = skeletonized_image[i-1:i+2, j-1:j+2]  # Extract 3x3 block
            if matches_pattern(block, patterns):
                node_positions.append((j, i))  # Add node position if pattern matches

    return node_positions

# Example usage with a skeletonized image
skeletonized_image = thinned_img_inverted/255
node_candidates = detect_nodes(skeletonized_image, patterns)
print("Detected nodes:", node_candidates)


plot_cluster_centers(skeletonized_image, node_candidates)

# Set DBSCAN parameters
eps = 5  # Maximum distance for points to be considered in the same cluster
min_samples = 1  # Minimum number of points required to form a cluster

# Perform clustering and get the cluster centers
cluster_centers, labels = cluster_nodes(node_candidates, eps=eps, min_samples=min_samples)

# Plot the cluster centers on the image
plot_cluster_centers(skeletonized_image, cluster_centers)
plot_cluster_centers(reduced_image, cluster_centers)

#%%
import cv2

# Function to convert pixel coordinates to real-world coordinates
def pixel_to_real_world(coord, scale_x, scale_y, padding_top, padding_left):
    x_pixel, y_pixel = coord
    x_real = dimensions[0] + (-padding_left + x_pixel) * scale_x
    y_real = dimensions[3] - (-padding_top + y_pixel) * scale_y
    return x_real, y_real

# Convert combined intersections back to the original image space
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in nodes]
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
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in nodes]

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
    #axes = (int(np.linalg.norm(np.array(node1) - np.array(center))), (int(np.linalg.norm(np.array(node1) - np.array(center)))/5) )  # semi-major and semi-minor axes
    axes = (
        int(np.linalg.norm(np.array(node1) - np.array(center))),  # Semi-major axis
        int(np.linalg.norm(np.array(node1) - np.array(center)) / 3)  # Semi-minor axis
    )
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
original_intersection_coordinates = [reverse_transformation(coord, transformation_rule) for coord in nodes]

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

