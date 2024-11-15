import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from system import System
from mesh import Mesh
from user_input import choose_system
import time


# TopOpt parameters
ft=0        # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
max_iteration = 50
mesh_ind_filter = True


# Import the System to optimize
from examples import create_mesh_cantilever, create_mesh_cantilever1, create_mesh_corbel, create_mesh_wall_with_openings, create_mesh_wall_without_openings, create_mesh_tower
from examples import create_mesh_bridge_2, create_mesh_bridge, create_mesh_bridge_1
s = create_mesh_cantilever1()


element_list = s.elements
node_list = s.nodes
x = s.x
x_min = s.x_min
r_min = s.r_min
volfrac = s.volfrac

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
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # possibility to define passive areas
        if 2<0: # for regular mesh only 
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

# The following must be initialized to use the NGuyen/Paulino OC approachgls
xold=x.copy()
xPhys=x.copy()
g=0 
obj_change = 1


# Initialize timing dictionary to store time values for each iteration
iteration_times = {
    'FE_solver': [],
    'Sensitivity_computation': [],
    'Filter': [],
    'Optimality_criteria': [],
    'Total_iteration': []
}

start_time_optim = time.time()
# while change>0.001 and loop<max_iteration: # original criteria from Sigmund
while obj_change > 0.00001 and loop < max_iteration:  # my own criteria
    start_time = time.time()
    loop = loop + 1

    # Solve FE problem
    fe_start_time = time.time()
    u = s.solve_FE_sparse()
    fe_end_time = time.time()
    iteration_times['FE_solver'].append(fe_end_time - fe_start_time)

    # Objective and sensitivity
    start_time_sens = time.time()
    obj = s.compliance()
    obj_hist.append(obj)
    if len(obj_hist) > 1:
        obj_change = abs(obj_hist[loop - 1] - obj_hist[loop - 2]) / obj_hist[loop - 1]
    # according to sigmund2001 eq4 (no filter)
    dc = s.sensitivity_compliance()
    end_time_sens = time.time()
    iteration_times['Sensitivity_computation'].append(end_time_sens - start_time_sens)

    # according to sigmund2001 eq5 (with filter)
    start_time_filt = time.time()
    if mesh_ind_filter:
        dc_filtered = []
        for i in range(len(element_list)):
            # additional if criteria compared to sigmund
            if x[i] * np.sum(H_f[:, i]) > 0:
                dc_filtered_i = 1 / x[i] * np.sum(H_f[:, i]) * np.sum(H_f[:, i] * x * dc)
            else:
                dc_filtered_i = dc[i]
            dc_filtered.append(dc_filtered_i)

        dc = dc_filtered

    end_time_filt = time.time()
    iteration_times['Filter'].append(end_time_filt - start_time_filt)

    dv = np.ones(len(element_list))
    # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens (not implemented)
    # if ft==0:
    #     dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
    # elif ft==1:
    #     dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
    #     dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]

    # Optimality criteria
    xold[:] = x
    start_time_oc = time.time()
    (x[:], g) = oc(len(element_list), x, volfrac, dc, dv, g)
    end_time_oc = time.time()
    iteration_times['Optimality_criteria'].append(end_time_oc - start_time_oc)

    # pass new x vector to system
    s.x = x

    # Filter design variables
    # if ft==0:   xPhys[:]=x
    # elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]

    # Compute the change by the inf. norm
    change = np.linalg.norm(x.reshape(len(element_list), 1) - xold.reshape(len(element_list), 1), np.inf)

    end_time = time.time()
    iteration_times['Total_iteration'].append(end_time - start_time)
    

    if (loop - 1) % 10 == 0:
        print('Iteration:', loop)
        print('obj:',obj)
        print('mean x:',np.mean(x))
        #s.plot2(deformed=False)
        #s.plot2(deformed=True)
        

end_time_optim = time.time()
print(f"total time iteration: {end_time_optim - start_time_optim:.6f} seconds \n")


# # Plotting the timing data for each iteration
# plt.figure(figsize=(12, 8))
# for key in iteration_times.keys():
#     plt.plot(iteration_times[key], label=key)

# plt.xlabel('Iteration')
# plt.ylabel('Time (seconds)')
# plt.title('Time Taken by Each Component Over Iterations')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



s.plot2(deformed=False)

# combined plot for obsidian
from image_processing_utils import combined_plot
combined_plot(s, obj_hist, x)


#%% processing and saving of image

from image_processing_utils import preprocess_image, save_preprocessed_image, save_plot_as_image, reduce_image_colors, plot_image, convert_to_binary, invert_image

# Generate and save the plot
plot_variable, dimensions = s.plot4(deformed=False, disp_bc=False, disp_corner=True)
original_image_path = save_plot_as_image(plot_variable, folder_name="C:/Users/luziu/Desktop/TO results/topopt_result")

# Preprocess the image
target_size = 256  
preprocessed_image = preprocess_image(original_image_path, target_size)

threshold = 150
# reduce image colors to black, white and red, green if disp_bc is True
reduced_image, dimensions_img = reduce_image_colors(preprocessed_image, grayscale_threshold=threshold, disp_bc=False)
plot_image(reduced_image)

# Save the preprocessed image 
preprocessed_image_path = save_preprocessed_image(reduced_image, folder_name="C:/Users/luziu/Desktop/TO results/preprocessed_images")

# binary image with structure in white
image_rgb = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)
binary_img = convert_to_binary(image_rgb)
image_inverted = invert_image(binary_img)

def save_binary(original_image_path, target_size=128):
    preprocessed_image = preprocess_image(original_image_path, target_size)
    binary_img = convert_to_binary(preprocessed_image)
    binary_image_path = save_preprocessed_image(binary_img, folder_name="C:/Users/luziu/Desktop/TO results/binary_images_128")

save_binary(original_image_path)


#%% BC nodes
from image_processing_utils import transformation_image_to_realworld, transformation_realworld_to_image

# boundary conditions could be passed on from the input in a usable format
# load_points = ([4,-1],[],)
# line_loads = ()
# fixed_points = ()
# fixed_lines = ([[0,-1],[0,1]],[])

# or need to be found again
from extraction_utils import process_supports_and_loads, transform_and_plot_bcs

# extract BCs from system
line_supports, line_loads, nodes_stm_bc = process_supports_and_loads(s)

# transform BCs to image space and plot BCs on topopt result
point_supports_img, line_supports_img, point_loads_img, line_loads_img = transform_and_plot_bcs(nodes_stm_bc, line_supports, line_loads, transformation_realworld_to_image, dimensions, dimensions_img, reduced_image)



#%% find nodes on support lines

from extraction_utils import cluster_nodes, plot_all_nodes, plot_cluster_centers, nodes_on_line_support

single_support_nodes_img = []
# find pixels on line supports with neighboring black pixels
if len(line_supports_img)>0:
    
    node_candidates = nodes_on_line_support(reduced_image, line_supports_img)
    
    # Perform clustering and get the cluster centers
    eps = 5  # Maximum distance for points to be considered in the same cluster
    min_samples = 3  # Minimum number of points required to form a cluster
    cluster_centers, labels = cluster_nodes(node_candidates, eps=eps, min_samples=min_samples)
    plot_all_nodes(reduced_image, node_candidates)
    plot_cluster_centers(reduced_image, cluster_centers, label='support points for STM')
    
    for coords in cluster_centers:
        single_support_nodes_img.append(coords)
    
  

#%% node detection with principal stresses
from extraction_utils import plot_principal_stresses, plot_tension_compression_zones, plot_nodal_zones_fang, plot_nodal_zones_alternative, plot_principal_stress_angles, cluster_and_plot

if 2 < 0:
    # Plot principal stresses
    plot_principal_stresses(element_list, x)
    
    # Plot tension and compression zones, and get average values
    sigma_t_avg, sigma_c_avg = plot_tension_compression_zones(element_list, x)
    
    # Plot nodal zones based on Fang2023 criteria
    plot_nodal_zones_fang(element_list, x, sigma_t_avg, sigma_c_avg)
    
    # Plot nodal zones based on alternative criteria
    plot_nodal_zones_alternative(element_list, x)
    
    # 3D principal forces angle plot
    plot_principal_stress_angles(element_list, x)
    
    # cluster the elemets based on their coordinates, principal stresses and directions using DBSCAN
    cluster_and_plot(element_list, x)

cluster_and_plot(element_list, x)
#%% extraction with my own node detection filter

# import utilities
from extraction_utils import cluster_nodes, plot_cluster_centers, find_node_candidates, plot_node_with_segments, plot_all_nodes

if 2<0:
    # set filter radius
    radius = 10 # minimum radius to start the search with
    min_angle_diff=10 # minimum opening angle for black and white segments
    # Run the node detection function
    
    node_candidates, segments_info, radii = find_node_candidates(reduced_image, radius=radius, min_angle_diff=np.deg2rad(min_angle_diff), white_threshold=0.05)
    
    # After detecting node candidates, call the function to plot them
    plot_all_nodes(reduced_image, node_candidates)
    
 
    # clustering
    
    # Set DBSCAN parameters
    eps = 5  # Maximum distance for points to be considered in the same cluster
    min_samples = 5  # Minimum number of points required to form a cluster
    
    # Perform clustering and get the cluster centers
    cluster_centers_filter, labels = cluster_nodes(node_candidates, eps=eps, min_samples=min_samples)
    
    # find the approximate size of the node
    # def get_node_radii(labels, node_candidates, radii):
    #     node_radii = np.zeros(max(labels)+1)
    #     cluster_size = np.zeros(max(labels)+1)
    #     for i in range(len(node_candidates)):
    #         if labels[i] >= 0:
    #             node_radii[labels[i]] += radii[i]
    #             cluster_size[labels[i]] += 1
    #     return node_radii / cluster_size
    
    # node_radii = get_node_radii(labels, node_candidates, radii)
    
    # Plot the cluster centers on the image
    plot_cluster_centers(reduced_image, cluster_centers_filter, label='internal nodes Filter')
    
    # Optionally, print the cluster centers
    print("Cluster Centers:", cluster_centers_filter)
    
    # visualizing segments
    
    # # Select a few or all node candidates to visualize the segments 
    # if node_candidates: 
    #     print('displaying a few node candidates together with the node detection filter')
    #     #for i in range(300,len(node_candidates)):
    #     for i in range(0,len(node_candidates),100):
    #         selected_node = node_candidates[i]
    #         segments = segments_info[selected_node]
            
    #         # Plot the circle and detected segments for the selected node
    #         plot_node_with_segments(reduced_image, selected_node, radius=radii[i], segments=segments)
            
    # else:
    #     print("No node candidates detected.")
            

#%% Node detection from Xia2020a with skeletonization from zhang1984
#skeletonization
from extraction_utils import zhang_suen_thinning, detect_nodes
# for the path following
from extraction_utils import generate_truss_structure_bfs_debug, plot_truss_structure, find_bcs_in_skeleton, update_truss_connections


if 2<4:
    # Apply the Zhang-Suen thinning algorithm on the inverted image
    if False:
        thinned_img_inverted = zhang_suen_thinning(image_inverted)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        plt.imshow(thinned_img, cmap='gray')
        plt.axis('off')  # Optional: turn off axis
        plt.show()
    
    
    # alternatively use the following library for skeletonization
    from skimage.morphology import skeletonize
    if True:
        skeleton = skeletonize(image_inverted > 0)
        thinned_img_inverted = (skeleton * 255).astype(np.uint8)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        plt.imshow(thinned_img, cmap='gray')
        plt.axis('off')  # Optional: turn off axis
        plt.show()


    # Node detection on skeletonized image
    skeletonized_image = thinned_img_inverted/255
    node_candidates = detect_nodes(skeletonized_image)
    print("Detected node candidates:", node_candidates)
    plot_cluster_centers(thinned_img, node_candidates, label='node candidates')
    
    
    # Optionally perform DBSCAN clustering and get the cluster centers
    cluster_centers_xia, labels = cluster_nodes(node_candidates, eps=3, min_samples=1)
    plot_cluster_centers(thinned_img, cluster_centers_xia, label='clustered nodes Xia')
    
    
    # Path following along the skeletonized image to detect trusses (connections)
    nodes_stm_bc_img = []
    for node in nodes_stm_bc:
        coords = node.coords
        nodes_stm_bc_img.append(transformation_realworld_to_image(coords, dimensions, dimensions_img))
        
    nodes_stm_bc_img.extend(single_support_nodes_img)
    nodes_skel_bc_img = find_bcs_in_skeleton(skeletonized_image, nodes_stm_bc_img)
    
    plot_cluster_centers(skeletonized_image, nodes_skel_bc_img, label='nodes_skel_bc_img')
    
    # add the nodes from the BCs 
    nodes_skel, nodes = [], []
    nodes_skel, nodes = node_candidates.copy(), node_candidates.copy()
    nodes_skel.extend(nodes_skel_bc_img.copy()), nodes.extend(nodes_stm_bc_img.copy())
    
    # plot all the nodes
    plot_cluster_centers(skeletonized_image, nodes_skel, label='all nodes')
    
    # Generate the truss structure
    truss_connections = generate_truss_structure_bfs_debug(nodes, nodes_skel, skeletonized_image)
    # Visualize the unique truss structure
    plot_truss_structure(skeletonized_image, truss_connections, nodes_skel)
    
    
    # Replace nodes skeleton BCs with actual BCs
    updated_connections = update_truss_connections(truss_connections, nodes_skel_bc_img, nodes_stm_bc_img)
    # Visualize the unique truss structure
    plot_truss_structure(reduced_image, updated_connections, nodes)
    


#%% computer vision approach
from extraction_utils import detect_intersections_and_lines_cv

if 2<0:
    cluster_centers_cv, lines = detect_intersections_and_lines_cv(image_inverted, reduced_image)

#%% back transformation to real world coords

from image_processing_utils import transformation_image_to_realworld
from node import Node

# Initialize nodes with cluster centers (cluster_centers_cv, cluster_centers_filter, cluster_centers_xia)
nodes_stm_internal_img = list(cluster_centers_xia)  # Ensure nodes is a list of tuples


for i, node in enumerate(nodes_stm_bc):
    # redistribute ids and dofs
    node.id = i
    node.dofs = [2*i, 2*i+1]

nodes_stm = nodes_stm_bc.copy()

# nodes from line supports
for coords in single_support_nodes_img:
    i+=1
    coords = transformation_image_to_realworld(coords, dimensions, dimensions_img)
    nodes_stm.append(Node(coords, i, [2*i, 2*i+1], fixed=[True, True]))

# transform coordinates to real world and generate node objects
nodes_stm_internal = []
for coords in nodes_stm_internal_img:
    i+=1
    coords = transformation_image_to_realworld(coords, dimensions, dimensions_img)
    nodes_stm.append(Node(coords, i, [2*i, 2*i+1]))


s.plot_fem_with_realworld_nodes(nodes_stm)


#%% find trusses (to be improved)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# def ellipse_mask(image_shape, center, axes, angle):
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
#     return mask

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

# def is_truss_between_nodes(image, node1, node2, nodes, threshold=0.75):
#     center = ((node1[0] + node2[0]) // 2, (node1[1] + node2[1]) // 2)
#     #axes = (int(np.linalg.norm(np.array(node1) - np.array(center))), (int(np.linalg.norm(np.array(node1) - np.array(center)))/5) )  # semi-major and semi-minor axes
#     axes = (
#         int(np.linalg.norm(np.array(node1) - np.array(center))),  # Semi-major axis
#         int(np.linalg.norm(np.array(node1) - np.array(center)) / 15)  # Semi-minor axis
#     )
#     angle = np.degrees(np.arctan2(node2[1] - node1[1], node2[0] - node1[0]))
    
#     mask = ellipse_mask(image.shape, center, axes, angle)
#     dark_pixel_count = np.sum(image[mask == 1] < 127)
#     total_pixel_count = np.sum(mask == 1)
    
#     # Check if any other node is inside the ellipse
#     for node in nodes:
#         if node != node1 and node != node2:
#             if is_node_in_ellipse(node, center, axes, angle):
#                 return False

#     return (dark_pixel_count / total_pixel_count) > threshold

def ellipse_mask(image_shape, center, axes, angle):
    """
    Create a binary mask with an ellipse on it.
    
    Parameters:
    - image_shape: Shape of the mask image (same as the original image).
    - center: Tuple representing the (x, y) coordinates of the ellipse center.
    - axes: Tuple representing the lengths of the semi-major and semi-minor axes.
    - angle: Rotation angle of the ellipse in degrees.
    
    Returns:
    - mask: Binary image with the ellipse filled in.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Create an empty mask
    center = (int(center[0]), int(center[1]))  # Ensure center is a tuple of ints
    axes = (int(axes[0]), int(axes[1]))  # Ensure axes are a tuple of ints
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)  # Draw filled ellipse
    return mask

def is_truss_between_nodes(image, node1, node2, nodes, threshold=0.45):
    """
    Determines if there is a truss between two nodes by checking the pixels between them.
    
    Parameters:
    - image: Grayscale image where darker pixels indicate possible truss paths.
    - node1, node2: Coordinates of the two nodes.
    - nodes: List of all node coordinates to check for intersections.
    - threshold: Minimum percentage of dark pixels needed to consider a truss.

    Returns:
    - True if a truss is found between node1 and node2, otherwise False.
    """
    # Calculate the center of the ellipse
    center = ((node1[0] + node2[0]) // 2, (node1[1] + node2[1]) // 2)
    
    # Calculate the axes (semi-major and semi-minor)
    semi_major_axis = int(np.linalg.norm(np.array(node1) - np.array(center)))
    semi_minor_axis = semi_major_axis // 5
    axes = (semi_major_axis, semi_minor_axis)
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(node2[1] - node1[1], node2[0] - node1[0]))
    
    # Generate the ellipse mask
    mask = ellipse_mask(image.shape, center, axes, angle)
    
    # Count dark pixels (assuming pixel value < 127 as "dark")
    dark_pixel_count = np.sum(image[mask == 1] < 127)
    total_pixel_count = np.sum(mask == 1)
    
    # Check if the mask contains any pixels
    if total_pixel_count == 0:
        return False  # No meaningful area to check, so no truss
    
    # Check if any other node is inside the ellipse
    for node in nodes:
        if node != node1 and node != node2:
            if is_node_in_ellipse(node, center, axes, angle):
                return False

    # Return True if the percentage of dark pixels exceeds the threshold
    return (dark_pixel_count / total_pixel_count) > threshold



# # Load the original image
# original_image = cv2.imread(preprocessed_image_path)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# nodes_stm_img = []
# for node in nodes_stm: 
#     coords = transformation_realworld_to_image(node.coords, dimensions, dimensions_img)
#     nodes_stm_img.append(coords)


# # Convert the combined intersections back to the original image space
# original_intersection_coordinates = nodes_stm_img

# # Convert original image to grayscale for truss detection
# gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# # Determine trusses between nodes
# trusses = []
# for i, node1 in enumerate(original_intersection_coordinates):
#     for j, node2 in enumerate(original_intersection_coordinates):
#         if i < j:
#             if is_truss_between_nodes(gray_original_image, node1, node2, original_intersection_coordinates):
#                 trusses.append((i, j))

# # Plot the original image with detected trusses
# plt.figure(figsize=(10, 10))
# plt.imshow(original_image)
# plt.title("Preprocessed Image with Detected Trusses and Intersection Points")

# # Overlay the intersection coordinates
# for idx, coord in enumerate(original_intersection_coordinates):
#     plt.scatter(*coord, color='blue', marker='x', s=100, label="Nodes")  # s is the size of the marker
#     plt.text(coord[0], coord[1], str(idx), color='blue', fontsize=30)

# # Overlay trusses
# for (i, j) in trusses:
#     node1 = original_intersection_coordinates[i]
#     node2 = original_intersection_coordinates[j]
#     plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='red', linewidth=2, label="Trusses")

# # Get the current legend handles and labels, and remove duplicates
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))  # Dictionary to remove duplicate labels

# # Plot settings
# plt.gca().set_aspect('equal', adjustable='box')
# plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
# plt.grid(True)
# plt.title("Strut and Tie Model (image space)")
# plt.show()



# # Optionally save the plotted image with intersections and trusses
# output_image_path_with_trusses = "original_image_with_trusses.png"

# print(f"Image with intersections and trusses saved to {output_image_path_with_trusses}")

# # Save node coordinates and truss elements to a CSV file
# csv_filename = "trusses_and_nodes.csv"
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write header for nodes
#     writer.writerow(["Node Index", "X Coordinate", "Y Coordinate"])
#     for idx, coord in enumerate(original_intersection_coordinates):
#         writer.writerow([idx + 1, coord[0], coord[1]])
    
#     # Write header for trusses
#     writer.writerow([])
#     writer.writerow(["Truss Index", "Start Node", "End Node"])
#     for truss_idx, (start_node, end_node) in enumerate(trusses):
#         writer.writerow([truss_idx + 1, start_node + 1, end_node + 1])

# print(f"CSV file with nodes and trusses saved to {csv_filename}")


#%% new idea
import hdbscan
from sklearn.preprocessing import StandardScaler
from extraction_utils import black_ratio


def not_in_node(center, nodes_stm, node_sizes):
    distances = np.zeros(len(nodes_stm))
    
    for i, node in enumerate(nodes_stm):
        distances[i] = np.sqrt((center[0] - node.coords[0])**2 + (center[1] - node.coords[1])**2)

    a = distances - node_sizes
    if min(a)<0:
        #print('in node')
        return False
    else:
        return True
    


def find_node_sizes(nodes_stm, image, dimensions, dimensions_img):
    # Ensure the image is in grayscale format
    if len(image.shape) == 3:  # If the image has 3 channels (e.g., RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    rows, cols = image.shape  # Ensure this is 2D
    
    node_radii = []
    for node in nodes_stm:
        coords = node.coords
        [x,y] = transformation_realworld_to_image(coords, dimensions, dimensions_img) 
        radius = 2
        ratio = black_ratio(image, x, y, radius)
        print(ratio)
        if ratio > 0.9:
            # print(ratio)
            while ratio>0.9 and radius < 100: # find the smallest circle around the current pixel that has approx 85% black pixels
                radius +=1
                ratio = black_ratio(image, x, y, radius)
                
        node_radii.append(radius)
        
    # Extract real-world dimensions
    (min_xs, min_ys), (max_xs, max_ys) = dimensions
    #print('dimensions', dimensions)
    
    # Extract image dimensions (in pixel coordinates)
    (bottom_left_x_img, bottom_left_y_img), (top_right_x_img, top_right_y_img) = dimensions_img
    #print('dimensions_img', dimensions_img)
    
    # Real-world to image scaling factors
    scale_x = (top_right_x_img - bottom_left_x_img) / (max_xs - min_xs)
    print(node_radii)
    
    node_radii_real = node_radii/scale_x
    
    return node_radii_real

def prepare_and_normalize_element_data(element_list, x, nodes_stm, node_sizes, alpha_threshold=1.3, x_filter=0.5):
    """
    Prepare data for principal stresses, angles, and center coordinates of elements.
    Normalize the data for clustering.

    Parameters:
    - element_list: List of elements to process.
    - x: List of densities used as a filter for elements.
    - alpha_threshold: Threshold for adjusting the angle `alpha`. Default is 1.3 radians.
    - x_filter: Filter for density values. Only elements where x > x_filter are considered.

    Returns:
    - elements: A NumPy array containing [sigma_1, sigma_2, alpha, center_x, center_y] for each element.
    - elements_scaled: Scaled version of the `elements` array (for clustering).
    """
    elements = []
    
    for i, e in enumerate(element_list):
        if x[i] > x_filter:  # Only process elements where x[i] > x_filter
            sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
            center = e.element_center()
            
               
            if  not_in_node(center, nodes_stm, node_sizes):
                # Adjust alpha if necessary
                if alpha > alpha_threshold:
                    alpha -= np.pi  # Adjust the angle alpha if it exceeds the threshold
    
                # Append the data [sigma_1, sigma_2, alpha, center_x, center_y] to elements
                # elements.append([sigma_1, sigma_2, alpha, center[0], center[1]])
                elements.append([center[0], center[1], alpha])
                
    # Convert elements to a NumPy array
    elements = np.array(elements)
    #print(elements.shape)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    elements_scaled = scaler.fit_transform(elements)

    return elements, elements_scaled

def cluster_and_plot_new(element_list, x, nodes_stm, node_sizes, min_cluster_size=1):
    """
    Apply HDBSCAN clustering on scaled data and plot the results using original coordinates.

    Parameters:
    - elements: Original data (with principal stresses, alpha, x, and y).
    - elements_scaled: Scaled version of the original data for clustering.
    - min_cluster_size: The minimum size of clusters to be considered valid.

    This function plots the clustering results.
    """
    
    # Step 1: Prepare and normalize the element data
    elements, elements_scaled = prepare_and_normalize_element_data(element_list, x, nodes_stm, node_sizes)

    # Step 2: Perform HDBSCAN clustering and plot
    
    # Apply HDBSCAN clustering on the scaled data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(elements_scaled)  # Get the cluster labels

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
        plt.scatter(xy[:, 0], xy[:, 1], c=[tuple(col)], label=f'Cluster {k}' if k != -1 else 'Noise', s=10)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'HDBSCAN Clustering in Original Space (min_cluster_size={min_cluster_size})')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.grid(True)
    plt.show()
    
    
node_sizes = find_node_sizes(nodes_stm, reduced_image, dimensions, dimensions_img)
cluster_and_plot_new(element_list, x, nodes_stm, node_sizes, min_cluster_size=10)



#%%

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import matplotlib.cm as cm
import hdbscan
from sklearn.preprocessing import StandardScaler
from extraction_utils import black_ratio

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
            if abs(sigma_2) > abs(sigma_1):
                alpha -= np.pi/2  # Adjust the angle alpha

            # Append the data [sigma_1, sigma_2, alpha, center_x, center_y] to elements
            elements.append([center[0], center[1], alpha, sigma_1, sigma_2])

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
    elements, elements_scaled = prepare_and_normalize_element_data(element_list, x)

    # Step 2: Perform HDBSCAN clustering and plot
    
    # Apply HDBSCAN clustering on the scaled data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(elements_scaled)  # Get the cluster labels

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
        plt.scatter(xy[:, 0], xy[:, 1], c=[tuple(col)], label=f'Cluster {k}' if k != -1 else 'Noise', s=3)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'HDBSCAN Clustering in Original Space (min_cluster_size={min_cluster_size})')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.grid(True)
    plt.show()
    
    return elements, labels
    




def cluster_direction_with_color(elements, labels, color_map):
    """
    Calculate the mean angle for each cluster and plot the angle distribution with consistent colors.
    
    Parameters:
    - elements: Array of element data, where each row represents an element's properties,
      including its orientation angle (`alpha`) in the third column.
    - labels: Array of cluster labels for each element, output from a clustering algorithm.
    - color_map: Dictionary mapping cluster labels to colors.
    
    Returns:
    - mean_angles: Dictionary with cluster label as key and the mean angle for that cluster as value.
    """
    
    # Find unique clusters, ignoring the noise label (-1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    # Initialize a dictionary to store mean angles for each cluster
    mean_angles = {}
    angle_distributions = {}

    for cluster_label in unique_labels:
        # Extract angles for elements in the current cluster
        cluster_angles = elements[labels == cluster_label, 2]  # Column 2 is the angle `alpha`
        mean_angle = np.mean(cluster_angles)
        
        # Store the mean angle in the dictionary
        mean_angles[cluster_label] = mean_angle
        angle_distributions[cluster_label] = cluster_angles

    # Plot angle distribution for each cluster
    plt.figure(figsize=(10, 6))
    for cluster_label, angles in angle_distributions.items():
        plt.hist(angles, bins=20, alpha=0.5, color=color_map[cluster_label], label=f'Cluster {cluster_label}')
    
    plt.xlabel('Angle (radians)')
    plt.ylabel('Frequency')
    plt.title('Angle Distribution per Cluster')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    return mean_angles


import numpy as np
import matplotlib.pyplot as plt

def create_color_map(labels):
    """
    Create a color map for clusters based on unique labels.
    
    Parameters:
    - labels: Array of cluster labels.
    
    Returns:
    - color_map: Dictionary mapping each cluster label to a color.
    """
    unique_labels = sorted(set(labels))
    colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    return color_map

def plot_clusters_with_mean_angle(elements, labels, mean_angles, color_map):
    """
    Plot clusters with points, and indicate each cluster's mean angle using a large black arrow.
    
    Parameters:
    - elements: Array of element data, where each row represents an element's properties,
      including its orientation angle (`alpha`) in the third column.
    - labels: Array of cluster labels for each element.
    - mean_angles: Dictionary with cluster label as key and the mean angle for that cluster as value.
    - color_map: Dictionary mapping cluster labels to colors.
    """
    plt.figure(figsize=(10, 8))
    
    for cluster_label, color in color_map.items():
        if cluster_label == -1:
            # Skip noise points
            continue

        # Get elements in the current cluster
        cluster_elements = elements[labels == cluster_label]
        coords = cluster_elements[:, 0:2]  # Columns 3 and 4 contain x and y coordinates

        # Plot points in the cluster
        plt.scatter(coords[:, 0], coords[:, 1], s=10, color=color, label=f'Cluster {cluster_label}')
        
        # Calculate mean center of the cluster
        mean_x, mean_y = np.mean(coords[:, 0]), np.mean(coords[:, 1])
        
        # Get the direction vector for the mean angle
        mean_angle = mean_angles[cluster_label]
        dx, dy = np.cos(mean_angle), np.sin(mean_angle)

        # Plot a large black arrow indicating the mean angle
        plt.arrow(mean_x, mean_y, dx * 10, dy * 10, color='black', width=0.2, 
                  head_width=1.5, head_length=1, fc='black', ec='black')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')
    plt.title('Clusters with Mean Angle Directions')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.show()


def find_and_plot_extreme_points(elements, labels, mean_angles, color_map):
    """
    Find and plot the two points furthest apart along the direction of the mean angle for each cluster,
    and draw a line connecting these two points.
    
    Parameters:
    - elements: Array containing [sigma_1, sigma_2, alpha, center_x, center_y] for each element.
    - labels: Array of cluster labels for each element.
    - mean_angles: Dictionary with cluster label as key and the mean angle for that cluster as value.
    - color_map: Dictionary mapping cluster labels to colors.
    
    Returns:
    - extreme_points: Dictionary where each key is a cluster label and value is a tuple with the two 
      furthest points' coordinates along the mean angle direction.
    """
    extreme_points = {}

    plt.figure(figsize=(10, 8))

    for cluster_label, mean_angle in mean_angles.items():
        color = color_map[cluster_label]
        # Get elements in the current cluster
        cluster_elements = elements[labels == cluster_label]
        
        # Extract x, y coordinates for these elements
        coords = cluster_elements[:, 0:2]  # Columns 3 and 4 contain x and y coordinates
        
        # Plot all points in the cluster
        plt.scatter(coords[:, 0], coords[:, 1], s=10, color=color, label=f'Cluster {cluster_label}')
        
        # Calculate the direction vector from the mean angle
        direction_vector = np.array([np.cos(mean_angle), np.sin(mean_angle)])
        
        # Project each point onto the direction vector
        projections = np.dot(coords, direction_vector)
        
        # Find the indices of the min and max projections
        min_index = np.argmin(projections)
        max_index = np.argmax(projections)
        
        # Get the two furthest points in the direction of the mean angle
        furthest_point_1 = coords[min_index]
        furthest_point_2 = coords[max_index]
        
        # Store these points in the dictionary
        extreme_points[cluster_label] = (furthest_point_1, furthest_point_2)
        
        # Plot the two extreme points with a connecting line
        plt.plot([furthest_point_1[0], furthest_point_2[0]],
                 [furthest_point_1[1], furthest_point_2[1]],
                 color=color, linestyle='-', linewidth=2)
        
        plt.scatter(*furthest_point_1, color=color, edgecolor='k', s=50, marker='o')
        plt.scatter(*furthest_point_2, color=color, edgecolor='k', s=50, marker='o')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X Center')
    plt.ylabel('Y Center')
    plt.title('Clusters with Extreme Points and Connecting Lines')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.show()
    
    return extreme_points


# Step 0: Cluster
elements, labels=cluster_and_plot(element_list, x)

# Step 1: Create a color map
color_map = create_color_map(labels)

# Step 2: Call each function with the color map
mean_angles = cluster_direction_with_color(elements, labels, color_map)
plot_clusters_with_mean_angle(elements, labels, mean_angles, color_map)
extreme_points = find_and_plot_extreme_points(elements, labels, mean_angles, color_map)
print("Extreme points for each cluster along the mean angle direction:", extreme_points)
