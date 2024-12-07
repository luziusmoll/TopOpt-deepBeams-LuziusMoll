import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import copy
import cv2


# TopOpt parameters
max_iteration = 50
mesh_ind_filter = True

# Define the path to save the results. If you dont want to save them set the path to None
path = "C:/Users/luziu/Documents/GitHub/TopOpt-deepBeams-LuziusMoll"

# Optimized systems used in the Thesis
from examples import create_mesh_cantilever0, create_mesh_cantilever1, create_mesh_corbel, create_mesh_wall_with_openings
# Other systems with non optimized parameters
from examples import  create_mesh_cantilever_short, create_mesh_cantilever1_hole

s = create_mesh_cantilever1_hole()


element_list = s.elements
node_list = s.nodes
x = s.x
x_min = s.x_min
r_min = s.r_min
volfrac = s.volfrac


example_name=s.name  

if mesh_ind_filter == False:
    example_name = f"{example_name}_no_filter"



#%% Convolution operator for mesh independency filtering
""" from sigmund2001: A 99 line topology optimization code written in Matlab: eq6"""


def convolution_operator(s):
    # distance between current element and all others
    element_centers = s.element_centers()
    element_centers = np.array(element_centers)
    
    dist = []
    for i in range(len(s.elements)):
        dist_ij = []
        for j in range(len(s.elements)):
            dist_x = element_centers[i,0]-element_centers[j,0]
            dist_y = element_centers[i,1]-element_centers[j,1]
            dist_ij.append(np.sqrt(dist_x**2 + dist_y**2))
        dist.append(dist_ij)
    
        
    # convolution operator H_f
    H_f = r_min * np.ones([len(s.x),len(s.x)]) - dist
    # set negativ values (elements outside of r_min) to zero
    H_f[H_f < 0] = 0
    
    return H_f


#%% Optimality criteria
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

def oc(x,volfrac,dc,dv):
    dc=np.array(dc)
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(len(x))
    while (l2-l1)/(l1+l2)>1e-8:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(x_min,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # possibility to define passive areas in regular mesh
        if 2<0: # for regular mesh only 
            for ely in range(40):
                for elx in range(80):
                    if np.sqrt((ely-20)**2 + (elx-30)**2) < 10:
                        xnew[elx*40+ely] = x_min
        
        # if np.mean(dv*xnew)> np.mean(dv*volfrac):
        if np.mean(xnew)> volfrac:   # this assumes that all elements have a comparable area. If that is not the case, a scaling with the element areas is necessary
            l1=lmid
        else:
            l2=lmid
            
        # with out this float division by 0 can occour in the while loop criteria (additional line compared to sigmund 200 line implementation)
        if l1 + l2 == 0:
            return xnew
        
    
    return xnew


#%% Actual optimization 
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """


def top_opt(s, x, H_f, dv, max_iteration):
    # Set loop counter and gradient vectors 
    loop=0
    obj_hist = []
    change=1

    # The following must be initialized to use the NGuyen/Paulino OC approach
    xold=x.copy()
    obj_change = 1

    
    
    while obj_change > 0.000001 and loop < max_iteration:  # my own criteria
        loop = loop + 1
    
        # Solve FE problem
        u = s.solve_FE_sparse()
        
        # Objective and sensitivity
        obj = s.compliance()
        obj_hist.append(obj)
        if len(obj_hist) > 1:
            obj_change = abs(obj_hist[loop - 1] - obj_hist[loop - 2]) / obj_hist[loop - 1]
        # according to sigmund2001 eq4 (no filter)
        dc = s.sensitivity_compliance()
        
        # according to sigmund2001 eq5 (with filter)
        if mesh_ind_filter:
            dc_filtered = []
            for i in range(len(s.elements)):
                # additional if criteria compared to sigmund
                if x[i] * np.sum(H_f[:, i]) > 0:
                    dc_filtered_i = 1 / x[i] * np.sum(H_f[:, i]) * np.sum(H_f[:, i] * x * dc)
                else:
                    dc_filtered_i = dc[i]
                dc_filtered.append(dc_filtered_i)
    
            dc = dc_filtered
    
        # Optimality criteria
        xold[:] = x
        x[:] = oc(x, volfrac, dc, dv)
    
        # pass new x vector to system
        s.x = x
    
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(len(s.elements), 1) - xold.reshape(len(s.elements), 1), np.inf)
    
        if (loop) % 5 == 0 or loop==1:
            print('Iteration:', loop)
            print('obj:',obj)
            print('mean x:',np.mean(x))
            s.plot2(deformed=False, disp_bc=False, line_thickness=0.2)    
    
    s.obj_hist = obj_hist
    
    # combined plot of optimized structure, objecitve history and element density distribution
    s.combined_plot()


# Run the optimization
dv = np.ones(len(s.elements))
H_f = convolution_operator(s)
top_opt(s, x, H_f, dv, max_iteration)



# save as pickle
if path is not None:
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # Define the file name
    full_path = os.path.join(path, f"{example_name}.pkl")
    
    # Save the object
    with open(full_path, "wb") as file:
        pickle.dump(s, file)
    
    print(f"System saved successfully to {full_path}")
    
    
#%% Alernatively load an already optimized system

if 2<0:
    # Define the path
    example_name = "wall_with_openings_N10814_r3_p3"
    full_path = os.path.join(path, f"systems/{example_name}.pkl")
    
    # Load the object
    with open(full_path, "rb") as file:
        s = pickle.load(file)
    
    print(f"System loaded successfully from {full_path}")
    


#%% processing and saving of image

from utils import preprocess_image, save_image, plot_image, convert_to_binary, invert_image

# Preprocess the image (reduce image colors to black, white and red, green if disp_bc is True)
target_size = 256 
grayscale_threshold = 102 
image, dimensions, dimensions_img = preprocess_image(s, os.path.join(path, "Results/TO Results"), target_size, grayscale_threshold=grayscale_threshold)

# plot and save the preprcessed image
plot_image(image)
save_image(image, os.path.join(path, f"Results/Preprocessed Images {target_size}/{s.name}.png"))

# binary image with structure in white
image_binary_inverted = invert_image(convert_to_binary(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
plot_image(image_binary_inverted)


#%% Initialize the strut and tie model object

from stm import STM

stm = STM(s, image, dimensions, dimensions_img)

# extract boundary conditions from system
stm.extract_bcs()


#%% node detection with principal stresses
from utils import plot_principal_stresses, plot_tension_compression_zones, plot_nodal_zones_fang, plot_nodal_zones_alternative, plot_principal_stress_angles, cluster_and_plot

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




#%% extraction with my own node detection filter

from utils import cluster_nodes, plot_cluster_centers, find_node_candidates, plot_node_with_segments

if 2<0:
    # set filter radius
    radius = 5 # minimum radius to start the search with
    min_angle_diff=10 # minimum opening angle for black and white segments
    
    # Run the node detection function
    node_candidates, segments_info, radii = find_node_candidates(image, radius=radius, min_angle_diff=np.deg2rad(min_angle_diff), white_threshold=0.05)
    
    # After detecting node candidates, call the function to plot them
    plot_cluster_centers(image, node_candidates, label='All node canditates Filter')
    
    # Perform DBSCAN clustering with parameters 
    # eps = Maximum distance for points to be considered in the same cluster
    # and min_samples = Minimum number of points required to form a cluster
    cluster_centers_filter, labels = cluster_nodes(node_candidates, eps=5, min_samples=5)
    plot_cluster_centers(image, cluster_centers_filter, label='internal nodes Filter')
    
    # Select a few or all node candidates to visualize the segments 
    if node_candidates: 
        print('displaying a few node candidates together with the node detection filter')
        #for i in range(300,len(node_candidates)):
        for i in range(0,len(node_candidates)):
            selected_node = node_candidates[i]
            segments = segments_info[selected_node]
            
            # Plot the circle and detected segments for the selected node
            plot_node_with_segments(image, selected_node, radius=radii[i], segments=segments,i=i)        
    else:
        print("No node candidates detected.")
            

#%% delauny

from scipy.spatial import Delaunay

if 2<0:
    cluster_centers_filter_bc = cluster_centers_filter.copy()
    for node in stm.node_list:
        cluster_centers_filter_bc.append(node.coords_img)
    
    points = np.array(cluster_centers_filter_bc)
    
    # Perform Delaunay triangulation
    tri = Delaunay(points)
    
    
    # Plot the flipped binary image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap='gray', interpolation='none')  # Use grayscale colormap
    ax.set_xlim([dimensions_img[0][0] - 5, dimensions_img[1][0] + 5])  # Add padding
    ax.set_ylim(dimensions_img[0][1] + 5, dimensions_img[1][1] - 5)  # Add padding
    
    
    
    line_width_mm = 1  # Line width in mm
    line_width_points = line_width_mm * 2.8346  # Convert mm to points
    
    plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue', linewidth=line_width_points)
    for node in stm.node_list:
        if np.any(node.forces != 0):
            plt.scatter(node.coords_img[0], node.coords_img[1], color='green', label='loads', marker='o', s=200,  zorder=2)
        elif any(node.fixed):
            plt.scatter(node.coords_img[0], node.coords_img[1], color='green', label='supports', marker='o', s=200, zorder=2)
    
    for coords in cluster_centers_filter:
        plt.scatter(coords[0], coords[1], color='blue', label='nodes', marker='o', s=200,  zorder=3)
    
    
    
    #plt.scatter(points[:, 0], points[:, 1], color=TUM_blue, marker='o', s=200, zorder=5)
    
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    # Adjust figure margins
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_tight_layout(False)
    plt.gcf().set_size_inches((8, 8), forward=True)  # Adjust size as needed
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.show()

#%% Skeletonization from zhang1984 and node detection from Xia2020a 
from utils import zhang_suen_thinning, detect_nodes,  generate_truss_structure_bfs, plot_cluster_centers , cluster_nodes


if 2<4:
    # Apply the customized Zhang-Suen thinning algorithm on the inverted image 
    # (pixels with boundary conditions are not removable)
    if True:
        thinned_img_inverted = zhang_suen_thinning(image_binary_inverted, stm)
        thinned_img = invert_image(thinned_img_inverted)
        skeletonized_image = thinned_img_inverted/255
        node_candidates = detect_nodes(skeletonized_image)
    
    
    # Alternatively use the following library for skeletonization 
    # pixels with boundary conditions can be deleted, which leads to an suboptimal mapping of the boundary conditions to the skeleton
    from skimage.morphology import skeletonize
    if False:
        thinned_img_inverted = (skeletonize(image_binary_inverted > 0) * 255).astype(np.uint8)
        thinned_img = invert_image(thinned_img_inverted)
        skeletonized_image = thinned_img_inverted/255
        node_candidates = detect_nodes(skeletonized_image)
        
        
    # Nodes that are directly next to each other are merged
    node_candidates =  cluster_nodes(node_candidates, eps=1.5, min_samples=1)
    node_candidates = node_candidates[0]
    
    # Plot the extracted nodes
    plot_cluster_centers(thinned_img, node_candidates, label='node candidates')
    
    # Add node candidates to stm
    stm.nodes_skel(skeletonized_image, node_candidates)
    
    # Generate the truss structure
    generate_truss_structure_bfs(stm, skeletonized_image)
    
    # Visualize the truss structure (on skeleton and image)
    stm.plot_truss_structure(thinned_img)
    

#%% computer vision approach

from utils import detect_intersections_and_lines_cv

if 2<0:
    cluster_centers_cv, lines = detect_intersections_and_lines_cv(image_binary_inverted, image)

#%% set up STM system in real world coords to allow for further analysis

# Plot of the extracted STM ontop of the TopOpt result
stm.plot_fem_with_realworld_nodes()

# Generate a new FE system for the STM
stm_system = stm.generate_stm_system()
   
# Solve the system 
u = stm_system.solve_FE()

# Plot the internal forces
stm_system.plot_internal_forces_stm()

# Plot the deformation
stm_system.plot_deformed_stm_sf(100, scale=100)

#%% set up design space and check if design space is violated by extracted stm

from shapeopt import shape_optimization, domain_penalty1


design_space = s.shapely_geometry
design_boundary = design_space[0]
if len(design_space)>1:
    holes = design_space[1:]
else:
    holes =None


fig, ax = plt.subplots(figsize=(8, 8))
p = domain_penalty1(stm_system, design_boundary, holes, node_weight=1.0, beam_weight=1.0, penalty_scale=1.0, ax=ax, plot=True)

if p>0:
    print('design space is violated.')
else:
    print('design space is respected.')

#%% shape opt

# System to optimize
system_shape_opt = copy.deepcopy(stm_system)


# Optimization paramters
N_iter = 100
penalty_ele = 0.05
l_B = 10
l_min = 0.1

# Modifiy stiffness values as needed
for e in system_shape_opt.elements:
    e.I = 0.01
    e.A = 1
    
    
# delete short elements
system_shape_opt.delete_short_elements(l_min)


# Optimization
shape_optimization(N_iter, system_shape_opt, design_boundary, holes, l_B=l_B,penalty_nodes=penalty_ele, penalty_ele=penalty_ele, domain_p_type=1,l_min=l_min)


# Plot deformation of optimized system
system_shape_opt.plot_deformed_stm_sf(100, scale=100)



