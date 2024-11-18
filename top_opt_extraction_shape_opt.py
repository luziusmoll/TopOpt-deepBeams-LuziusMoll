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
from examples import create_mesh_cantilever, create_mesh_cantilever1, create_mesh_cantilever2, create_mesh_corbel, create_mesh_wall_with_openings, create_mesh_wall_without_openings, create_mesh_tower
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


def convolution_operator(s):
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
    H_f = r_min * np.ones([len(s.x),len(s.x)]) - dist
    # set negativ values (elements outside of r_min) to zero
    H_f[H_f < 0] = 0
    
    return H_f

H_f = convolution_operator(s)
        
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


def top_opt(s, x, H_f, max_iteration):
    # Set loop counter and gradient vectors 
    loop=0
    obj_hist = []
    change=1

    # The following must be initialized to use the NGuyen/Paulino OC approach
    xold=x.copy()
    # xPhys=x.copy()
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
        #x = s.x.copy()
    
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


    # s.plot2(deformed=False)
    
    # combined plot for obsidian
    from image_processing_utils import combined_plot
    combined_plot(s, obj_hist, x)
    
    
    return iteration_times

iteration_times = top_opt(s, x, H_f, max_iteration)


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
from extraction_utils import generate_truss_structure_bfs, plot_cluster_centers #, find_bcs_in_skeleton, update_truss_connections


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

#%% set up STM in real world coords to allow for shapeOpt

stm.plot_fem_with_realworld_nodes()


stm_system = stm.generate_stm_system()


# solve the system 
u = stm_system.solve_FE()

# plot the stm and its displacements
stm_system.plot_deformation_stm(scale=100)




stm_system.plot_internal_forces_stm()




# # Stablängskräfte aller Elemente in einen vektor N_i = []
# N_i = stm_system.Rückrechnung_stablängskraft(1)
# # print('Normalkraft nach Th. 1. O.:', N_i)

# M_rand = stm_system.Rückrechnung_randmomente()

stm_system.plot_deformed_stm(100, scale=100)

#%%
import copy

system_shape_opt = copy.deepcopy(stm_system)

for e in system_shape_opt.elements:
    e.EI = 0.01

# Compute dimensions
min_coords = [float('inf'), float('inf')]  # [min_x, min_y]
max_coords = [-float('inf'), -float('inf')]  # [max_x, max_y]

for node in system_shape_opt.nodes:
    x, y = node.coords
    if x < min_coords[0]:
        min_coords[0] = x  # Update min_x
    if y < min_coords[1]:
        min_coords[1] = y  # Update min_y
    if x > max_coords[0]:
        max_coords[0] = x  # Update max_x
    if y > max_coords[1]:
        max_coords[1] = y  # Update max_y

dimension = [max_coords[0] - min_coords[0], max_coords[1] - min_coords[1]]  # [width, height]
dx = dimension[0] / 1000
dy = dimension[1] / 1000

# Sensitivity analysis
compliance = system_shape_opt.compliance()
d_c = np.zeros(len(node_list) * 2)  # Assuming 2 DOFs per node


# Optimization parameters
max_iter = 200  # Maximum number of iterations
tolerance = 1e-10  # Convergence tolerance for compliance
move_limit_x = dx  # Maximum allowable change in x-direction
move_limit_y = dy  # Maximum allowable change in y-direction
eta = (dx + dy) / (max(d_c) + dx) # step size

iteration = 0
compliance_prev = float('inf')  # Initialize previous compliance to a large value
compliance_history = []  # To store compliance values over iterations

while iteration < max_iter:

    # Step 1: Compute compliance and sensitivity
    compliance = system_shape_opt.compliance()
    compliance_history.append(compliance)  # Store current compliance value
    d_c = np.zeros(len(node_list) * 2)  # Sensitivity array for x and y coordinates

    for i, node in enumerate(system_shape_opt.nodes):
        # Skip fixed nodes
        if any(node.fixed):
            d_c[i * 2] = 0
            d_c[i * 2 + 1] = 0
            # print(f"Node {i} fixed")
            continue

        # Skip nodes with non-zero external forces
        if np.linalg.norm(node.forces) > 0:  # Check if forces are non-zero
            d_c[i * 2] = 0
            d_c[i * 2 + 1] = 0
            # print(f"Node {i} has external forces")
            continue

        # Compute sensitivity for internal nodes
        for coord_index in range(2):  # x and y coordinates
            original_value = node.coords[coord_index]
            node.coords[coord_index] += dx  # Perturb the coordinate
            system_shape_opt.solve_FE()  # Recalculate system with perturbed geometry
            compliance_var = system_shape_opt.compliance()
            d_c[i * 2 + coord_index] = (compliance_var - compliance) / dx
            node.coords[coord_index] = original_value  # Reset to original

    # Step 2: Update nodal coordinates in the negative d_c direction
    # Update nodal coordinates with capped step size
    for i, node in enumerate(system_shape_opt.nodes):
        if any(node.fixed) or np.linalg.norm(node.forces) > 0:
            continue  # Skip fixed or loaded nodes
    
        # Compute step size for x and y directions
        step_x = max(min(eta * d_c[i * 2], move_limit_x), -move_limit_x)  # Cap step size by move_limit_x
        step_y = max(min(eta * d_c[i * 2 + 1], move_limit_y), -move_limit_y)  # Cap step size by move_limit_y
    
        # Update coordinates while staying within bounds
        node.coords[0] = max(min(node.coords[0] - step_x, max_coords[0]), min_coords[0])
        node.coords[1] = max(min(node.coords[1] - step_y, max_coords[1]), min_coords[1])


    # Step 3: Check convergence
    compliance_change = abs(compliance_prev - compliance)

    if compliance_change < tolerance:
        print("Convergence achieved!")
        break

    # Optional: Plot the deformed structure at each iteration
    if iteration % 20 ==0:
        print(f"Iteration {iteration + 1}")
        system_shape_opt.plot_deformed_stm(100, scale=10, title=f'Iteration: {iteration}')
        print(f"Compliance: {compliance}, Change: {compliance_change}")
    
    
    # Update previous compliance and iteration counter
    compliance_prev = compliance
    iteration += 1

# Final output
print("Optimization completed.")
print(f"Final Compliance: {compliance}")

# Plot the compliance history
plt.figure(figsize=(8, 6))
plt.plot(compliance_history, label="Compliance History", marker="o")
plt.xlabel("Iteration")
plt.ylabel("Compliance")
plt.title("Compliance History During Optimization")
plt.grid(True)
plt.legend()
plt.show()


# plot the internal forces
system_shape_opt.plot_internal_forces_stm()


# calculate ratio of normal forces
sts = system_shape_opt.sts()
print('sts per element')
print(sts)
print('sts:')
print(np.mean(sts))




