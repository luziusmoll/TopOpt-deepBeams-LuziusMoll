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
    print(f"total time TopOpt: {end_time_optim - start_time_optim:.6f} seconds \n")
    
    
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
image, dimensions, dimensions_img = preprocess_image(s, path, target_size, grayscale_threshold=102, disp_bc=False)


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
    if True:
        thinned_img_inverted = zhang_suen_thinning(image_binary_inverted)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        # plt.imshow(thinned_img, cmap='gray')
        # plt.axis('off')  # Optional: turn off axis
        # plt.show()
        # Node detection on skeletonized image
        skeletonized_image = thinned_img_inverted/255
        node_candidates = detect_nodes(skeletonized_image)
    
    
    # alternatively use the following library for skeletonization
    from skimage.morphology import skeletonize
    if False:
        thinned_img_inverted = (skeletonize(image_binary_inverted > 0) * 255).astype(np.uint8)
        # Invert the thinned image back to the original format
        thinned_img = invert_image(thinned_img_inverted)
        # Display the final thinned image
        # plt.imshow(thinned_img, cmap='gray')
        # plt.axis('off')  # Optional: turn off axis
        # plt.show()
        # Node detection on skeletonized image
        skeletonized_image = thinned_img_inverted/255
        node_candidates = detect_nodes(skeletonized_image, match='exact')


    
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

#%% set up STM system in real world coords to allow for shapeOpt

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

stm_system.plot_deformed_stm_sf(100, scale=100)


#%% shape opt
import copy
from shapely.geometry import Point, LineString, Polygon

# System to optimize
system_shape_opt = copy.deepcopy(stm_system)

# Define the design boundary
design_boundary = Polygon([
    [0.0, -1.0], [4.0, -1.0], [4.0, 1.0], [0.0, 1.0]
])

# Define a hole if present
hole = Polygon([
    [1.0, 0.2], [1.5, 0.2], [1.5, -0.2], [1.0, -0.2]
])

holes = [hole]
holes = []

for e in system_shape_opt.elements:
    e.I = e.A*0.001
    
    
# delete short elements
system_shape_opt.delete_short_elements(0.1)
system_shape_opt.plot_deformed_stm_sf(100, scale=100)


from shapeopt import shape_optimization

    
shape_optimization(system_shape_opt, design_boundary, holes, penalty_nodes=1, penalty_ele=1, domain_p_type=1)


# calculate ratio of normal forces
sts = system_shape_opt.sts()

formatted_sts = [f"{value[0]:.4f}" for value in sts]
print("STS per Element:", ", ".join(formatted_sts))

print('sts:')
print(np.mean(sts))

#%% Check if nodes or elements are outside the design space

from shapely.geometry import Point, LineString, Polygon


# Check nodes
for node in system_shape_opt.nodes:
    point = Point(node.coords)
    if not design_boundary.covers(point):  # Use covers() instead of contains()
        print(f"Node {node.id} is outside the design boundary")
    elif hole.contains(point):  # `contains` is fine here for holes
        print(f"Node {node.id} is inside a hole")
        
# Check beams
for element in system_shape_opt.elements:
    start_node, end_node = element.nodes
    beam_segment = LineString([start_node.coords, end_node.coords])

    # Check if beam is inside the design boundary
    if not design_boundary.covers(beam_segment):  # Use covers() here as well
        print(f"Beam {element.id} crosses outside the design boundary")

    # Check if beam crosses a hole
    if beam_segment.intersects(hole):
        print(f"Beam {element.id} crosses a hole boundary")




#%% Shape Optimization with penalty for design space enforcement
# from shapeopt import shape_optimization

    
# shape_optimization(system_shape_opt, design_boundary, holes, penalty_nodes=1, penalty_ele=1, domain_p_type=1)


# # calculate ratio of normal forces
# sts = system_shape_opt.sts()

# formatted_sts = [f"{value[0]:.4f}" for value in sts]
# print("STS per Element:", ", ".join(formatted_sts))

# print('sts:')
# print(np.mean(sts))


#%% calfem -> shapely
# import calfem.geometry as cfg
# import calfem.mesh as cfm

# g = cfg.Geometry()

# g.point([0.0, 0.0], ID=0) # point 0
# g.point([122.5, 0.0], ID=1) # point 1
# g.point([122.5, 75.0], ID=2) # point 2
# g.point([0.0, 75.0], ID=3) # point 3

# # opwning 1
# g.point([12.5, 30.0], ID=4)
# g.point([27.5, 30.0], ID=5)
# g.point([27.5, 45.0], ID=6)
# g.point([12.5, 45.0], ID=7)

# # opening 2
# g.point([95, 30.0], ID=8)
# g.point([110, 30.0], ID=9)
# g.point([110, 45.0], ID=10)
# g.point([95, 45.0], ID=11)


# g.spline([0, 1], ID=0) # line 0
# g.spline([1, 2], ID=1) # line 1
# g.spline([2, 3], ID=2) # line 2
# g.spline([3, 0], ID=3) # line 3

# g.spline([4, 5], ID=4)
# g.spline([5, 6], ID=5)
# g.spline([6, 7], ID=6)
# g.spline([7, 4], ID=7)

# g.spline([8, 9], ID=8)
# g.spline([9, 10], ID=9)
# g.spline([10, 11], ID=10)
# g.spline([11, 8], ID=11)


# g.surface([0, 1, 2, 3], [[4,5,6,7],[8,9,10,11]])



# mesh = cfm.GmshMesh(g)

# mesh.elType = 3 
# mesh.dofsPerNode = 2     
# mesh.elSizeFactor = 3

# coords, edof, dofs, bdofs, elementmarkers = mesh.create()

# # import pycalfem_vis as pcv
# # pcv.drawGeometry(g)


# # boundary_nodes = np.unique(np.array(bdofs[0]) // 2)



# from shapely.geometry import Polygon, LinearRing, Point
# import numpy as np
# from scipy.spatial import distance_matrix

# def extract_boundaries_from_mesh(coords, bdofs, tol=1e-6):
#     """
#     Extract the design boundary and holes from the mesh based on boundary DOFs.

#     Parameters:
#     - coords: Array of node coordinates (N x 2, where N is the number of nodes).
#     - bdofs: Dictionary containing lists of DOFs for each boundary marker.
#     - tol: Distance tolerance for identifying connected components.

#     Returns:
#     - design_boundary: Shapely Polygon representing the outer boundary.
#     - holes: List of Shapely Polygons representing holes.
#     """
#     if 0 not in bdofs:
#         raise ValueError("No boundary DOFs found in the bdofs dictionary.")

#     # Extract boundary nodes
#     boundary_nodes = np.unique(np.array(bdofs[0]) // 2)
#     boundary_coords = [coords[node] for node in boundary_nodes]

#     # Calculate pairwise distances between boundary nodes
#     dist_matrix = distance_matrix(boundary_coords, boundary_coords)

#     # Group boundary nodes into connected components
#     visited = set()
#     components = []

#     def dfs(node, component):
#         """Depth-first search to find connected nodes."""
#         visited.add(node)
#         component.append(node)
#         for neighbor, dist in enumerate(dist_matrix[node]):
#             if neighbor not in visited and dist < tol:
#                 dfs(neighbor, component)

#     for i in range(len(boundary_coords)):
#         if i not in visited:
#             component = []
#             dfs(i, component)
#             components.append(component)

#     # Construct polygons from connected components
#     polygons = []
#     for component in components:
#         coords_component = [boundary_coords[i] for i in component]
#         if len(coords_component) > 2:
#             ring = LinearRing(coords_component)
#             if ring.is_valid:
#                 polygons.append(Polygon(ring))

#     # Sort polygons by area (largest is the outer boundary)
#     polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
#     if len(polygons) == 0:
#         raise ValueError("No valid polygons found in the mesh boundaries.")

#     design_boundary = polygons[0]  # Largest polygon is the outer boundary
#     holes = polygons[1:]  # Remaining polygons are holes

#     return design_boundary, holes


# import matplotlib.pyplot as plt

# design_boundary, holes = extract_boundaries_from_mesh(coords, bdofs)

# # Plot design boundary
# x, y = design_boundary.exterior.xy
# plt.plot(x, y, 'blue', label='Design Boundary')

# # Plot holes
# for hole in holes:
#     x, y = hole.exterior.xy
#     plt.plot(x, y, 'red', linestyle='--', label='Hole')

# plt.title("Extracted Boundaries")
# plt.legend()
# plt.grid(True)
# plt.show()
