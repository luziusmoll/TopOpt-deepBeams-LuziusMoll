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

from image_processing_utils import preprocess_image, save_image, plot_image, convert_to_binary, invert_image
import os


# path were the results are saved
path = "C:/Users/luziu/Desktop/TO results"
 

# Preprocess the image (reduce image colors to black, white and red, green if disp_bc is True)
target_size = 256 
image, dimensions, dimensions_img = preprocess_image(s, path, target_size, grayscale_threshold=150, disp_bc=False)


# plot and save the preprcessed image
plot_image(image)
save_image(image, folder_name=os.path.join(path, f"preprocessed_images_{target_size}"))


# binary image with structure in white
image_binary_inverted = invert_image(convert_to_binary(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
plot_image(image_binary_inverted)


#%% initialize the strut and tie model
from stm import STM

stm = STM(s, image, dimensions, dimensions_img)

# extract boundary conditions from system
stm.extract_bcs()


# alternatively boundary conditions could be passed on from the input in a usable format
# load_points = ([4,-1],[],)
# line_loads = ()
# fixed_points = ()
# fixed_lines = ([[0,-1],[0,1]],[])


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
    
    node_candidates, segments_info, radii = find_node_candidates(image, radius=radius, min_angle_diff=np.deg2rad(min_angle_diff), white_threshold=0.05)
    
    # After detecting node candidates, call the function to plot them
    plot_all_nodes(image, node_candidates)
    
 
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
    plot_cluster_centers(image, cluster_centers_filter, label='internal nodes Filter')
    
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
from extraction_utils import generate_truss_structure_bfs #, find_bcs_in_skeleton, update_truss_connections


if 2<4:
    # Apply the Zhang-Suen thinning algorithm on the inverted image
    if False:
        thinned_img_inverted = zhang_suen_thinning(image_binary_inverted)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        # plt.imshow(thinned_img, cmap='gray')
        # plt.axis('off')  # Optional: turn off axis
        # plt.show()
    
    
    # alternatively use the following library for skeletonization
    from skimage.morphology import skeletonize
    if True:
        thinned_img_inverted = (skeletonize(image_binary_inverted > 0) * 255).astype(np.uint8)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        # plt.imshow(thinned_img, cmap='gray')
        # plt.axis('off')  # Optional: turn off axis
        # plt.show()


    # Node detection on skeletonized image
    skeletonized_image = thinned_img_inverted/255
    node_candidates = detect_nodes(skeletonized_image)
    print("Detected node candidates:", node_candidates)
    plot_cluster_centers(thinned_img, node_candidates, label='node candidates')
    
    
    # Optionally perform DBSCAN clustering and get the cluster centers
    # cluster_centers_xia, labels = cluster_nodes(node_candidates, eps=3, min_samples=1)
    # plot_cluster_centers(thinned_img, cluster_centers_xia, label='clustered nodes Xia')
    
    
    # add node candidates to stm
    stm.nodes_skel(skeletonized_image, node_candidates)
    

    # Generate the truss structure
    generate_truss_structure_bfs(stm, skeletonized_image)
    
    
    # Visualize the unique truss structure (on skeleton and image)
    stm.plot_truss_structure(thinned_img)
    



#%% computer vision approach
from extraction_utils import detect_intersections_and_lines_cv

if 2<0:
    cluster_centers_cv, lines = detect_intersections_and_lines_cv(image_binary_inverted, image)

#%% back transformation to real world coords

# from image_processing_utils import transformation_image_to_realworld
# from node import Node

# # Initialize nodes with cluster centers (cluster_centers_cv, cluster_centers_filter, cluster_centers_xia)
# nodes_stm_internal_img = list(cluster_centers_xia)  # Ensure nodes is a list of tuples


# for i, node in enumerate(nodes_stm_bc):
#     # redistribute ids and dofs
#     node.id = i
#     node.dofs = [2*i, 2*i+1]

# nodes_stm = nodes_stm_bc.copy()

# # nodes from line supports
# for coords in single_support_nodes_img:
#     i+=1
#     coords = transformation_image_to_realworld(coords, dimensions, dimensions_img)
#     nodes_stm.append(Node(coords, i, [2*i, 2*i+1], fixed=[True, True]))

# # transform coordinates to real world and generate node objects
# nodes_stm_internal = []
# for coords in nodes_stm_internal_img:
#     i+=1
#     coords = transformation_image_to_realworld(coords, dimensions, dimensions_img)
#     nodes_stm.append(Node(coords, i, [2*i, 2*i+1]))


# s.plot_fem_with_realworld_nodes(nodes_stm)

#%% set up STM to allow for shapeOpt








