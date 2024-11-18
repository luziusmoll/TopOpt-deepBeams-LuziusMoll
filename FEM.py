import numpy as np
import matplotlib.pyplot as plt
from system import System
from mesh import Mesh
import time

# Import the System to solve
from examples import create_mesh_cantilever, create_mesh_corbel, create_mesh_wall_with_openings, create_mesh_wall_without_openings, create_mesh_tower
from examples import create_mesh_bridge_2, create_mesh_bridge, create_mesh_bridge_1
s = create_mesh_cantilever()



# solve for initial x vector
u = s.solve_FE()
u = s.solve_FE_taichi()
u = s.solve_FE_sparse()
# Measure the performance of solve_FE_sparse
# start_time = time.time()
# u = s.solve_FE_taichi()
# end_time = time.time()
# print(f"solve_FE_sparse time: {end_time - start_time:.6f} seconds")

obj = s.compliance()
dc = s.sensitivity_compliance()
s.plot(deformed=False)
s.plot(deformed=True)

#%% comparison of FEM solvers
if 2<0:
    # Measure the performance of solve_FE
    print('solve_FE:')
    start_time = time.time()
    u = s.solve_FE()
    end_time = time.time()
    print(f"solve_FE total time: {end_time - start_time:.6f} seconds \n")
    
    # Measure the performance of solve_FE_sparse
    print('solve_FE_sparse:')
    start_time = time.time()
    u = s.solve_FE_sparse()
    end_time = time.time()
    print(f"solve_FE_sparse total time: {end_time - start_time:.6f} seconds\n")
    
    # Measure the performance of solve_FE_taichi
    print('solve_FE_taichi:')
    start_time = time.time()
    u = s.solve_FE_taichi()
    end_time = time.time()
    print(f"solve_FE_taichi total time: {end_time - start_time:.6f} seconds")
    



#%% test of single beam elements
from node import Node
from beam_element import BeamElement
import numpy as np
from system import System



  
node1 = Node([1,0], 0, [0,1,2], fixed=[True, True, True], forces=[0,0,0])
node2 = Node([-1,0], 1, [3,4,5], fixed=[False, False, False], forces=[0,-10,0])
    
node_list = [node1, node2]
 
for node in node_list:
    node.displacements = np.zeros(3)
  

beam1 = BeamElement(0, [node1,node2])


element_list = [beam1]

        
x = np.ones((len(element_list))) 


system_canti = System(node_list, element_list, x)

# apply dirichlet BCs
system_canti.apply_dirichlet_bc()


u = system_canti.solve_FE()

# plot the stm and its displacements (linear interpolation)
system_canti.plot_deformation_stm(scale=10)


system_canti.plot_internal_forces_stm()

# plot the stm and its displacements (interpolation using the shape functions)
system_canti.plot_deformed_stm(100, scale=10)


#%% test of system with beam elements
from node import Node
from beam_element import BeamElement

  
# node1 = Node([0,1], 0, [0,1,2], fixed=[True, True, False], forces=[0,0,0])
# node2 = Node([0,-1], 1, [3,4,5], fixed=[True, True, False], forces=[0,0,0])
# node3 = Node([1,0.5], 0, [0,1,2], fixed=[False, False, False], forces=[0,0,0])
# node4 = Node([1,-0.5], 1, [3,4,5], fixed=[False, False, False], forces=[0,0,0])
# node5 = Node([2,0.5], 0, [0,1,2], fixed=[False, False, False], forces=[0,0,0])
# node6 = Node([2,-0.5], 1, [3,4,5], fixed=[False, False, False], forces=[0,0,0])
# node7 = Node([1.5,0], 2, [6,7,8], fixed=[False, False, False], forces=[0,0,0])
# node8 = Node([4,0], 2, [6,7,8], fixed=[False, False, False], forces=[0,-1,0])


node1 = Node([0,1], 0, [0,1,2], fixed=[True, True, False], forces=[0,0,0])
node2 = Node([0,-1], 1, [3,4,5], fixed=[True, True, False], forces=[0,0,0])
node3 = Node([1,2], 0, [0,1,2], fixed=[False, False, False], forces=[0,0,0])
node4 = Node([1,-2], 1, [3,4,5], fixed=[False, False, False], forces=[0,0,0])
node5 = Node([2,2], 0, [0,1,2], fixed=[False, False, False], forces=[0,0,0])
node6 = Node([2,-2], 1, [3,4,5], fixed=[False, False, False], forces=[0,0,0])
node7 = Node([1.5,0], 2, [6,7,8], fixed=[False, False, False], forces=[0,0,0])
node8 = Node([4,0], 2, [6,7,8], fixed=[False, False, False], forces=[0,-1,0])
    
node_list = [node1, node2, node3, node4, node5, node6, node7, node8]
 
for i, node in enumerate(node_list):
    node.displacements = np.zeros(3)
    node.id = i
    node.dofs = [i*3, i*3+1,i*3+2]
    
  

beam1 = BeamElement(0, [node1, node3])
beam2 = BeamElement(1, [node2, node4])
beam3 = BeamElement(2, [node3, node5])
beam4 = BeamElement(3, [node5, node8])
beam5 = BeamElement(4, [node4, node6])
beam6 = BeamElement(5, [node6, node8])
beam7 = BeamElement(6, [node4, node7])
beam8 = BeamElement(7, [node6, node7])
beam9 = BeamElement(8, [node3, node7])
beam10 = BeamElement(9, [node5, node7])

# Add all beams to the element_list
element_list = [beam1, beam2, beam3, beam4, beam5, beam6, beam7, beam8, beam9, beam10]

for e in element_list:
    e.I = 0.01

# Density of the elements  
x = np.ones((len(element_list))) 

# Create system
system_stm = System(node_list, element_list, x)

# apply dirichlet BCs
system_stm.apply_dirichlet_bc()

# solve system
u = system_stm.solve_FE()

# plot the stm and its displacements (linear interpolation)
system_stm.plot_deformation_stm(scale=10)

# plot the internal forces
system_stm.plot_internal_forces_stm()

# plot the deformed stm using the shape functions of the beam elements
system_stm.plot_deformed_stm(100, scale=10)



# optimized compliance by varying nodal positions 

# Compute dimensions
min_coords = [float('inf'), float('inf')]  # [min_x, min_y]
max_coords = [-float('inf'), -float('inf')]  # [max_x, max_y]

for node in node_list:
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
compliance = system_stm.compliance()
d_c = np.zeros(len(node_list) * 2)  # Assuming 2 DOFs per node

for i, node in enumerate(system_stm.nodes):
    # if fixed, dont vary
    if any(node.fixed):
        d_c[i * 2] = 0
        d_c[i * 2 + 1] = 0
        print(f'Node {i} fixed')
        continue

    # Skip nodes with non-zero external forces
    if np.linalg.norm(node.forces) > 0:  # Check if forces are non-zero
        d_c[i * 2] = 0
        d_c[i * 2 + 1] = 0
        print(f'Node {i} has external forces')
        continue
    for coord_index in range(2):  # Loop over x and y coordinates
        original_value = node.coords[coord_index]
        node.coords[coord_index] += dx
        u = system_stm.solve_FE()  # Ensure stiffness matrices are recalculated
        compliance_var = system_stm.compliance()
        d_c[i * 2 + coord_index] = (compliance_var - compliance) / dx
        node.coords[coord_index] = original_value  # Reset to original

# Output sensitivity results
print("Sensitivity d_c:", d_c)



# shape optimization

# Optimization parameters
max_iter = 200  # Maximum number of iterations
tolerance = 1e-8  # Convergence tolerance for compliance
move_limit_x = dx  # Maximum allowable change in x-direction
move_limit_y = dy  # Maximum allowable change in y-direction
eta = (dx + dy) / (max(d_c) + dx) # step size

iteration = 0
compliance_prev = float('inf')  # Initialize previous compliance to a large value
compliance_history = []  # To store compliance values over iterations

while iteration < max_iter:
    print(f"Iteration {iteration + 1}")

    # Step 1: Compute compliance and sensitivity
    compliance = system_stm.compliance()
    compliance_history.append(compliance)  # Store current compliance value
    d_c = np.zeros(len(node_list) * 2)  # Sensitivity array for x and y coordinates

    for i, node in enumerate(system_stm.nodes):
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
            system_stm.solve_FE()  # Recalculate system with perturbed geometry
            compliance_var = system_stm.compliance()
            d_c[i * 2 + coord_index] = (compliance_var - compliance) / dx
            node.coords[coord_index] = original_value  # Reset to original

    # Step 2: Update nodal coordinates in the negative d_c direction
    # Update nodal coordinates with capped step size
    for i, node in enumerate(system_stm.nodes):
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
    print(f"Compliance: {compliance}, Change: {compliance_change}")

    if compliance_change < tolerance:
        print("Convergence achieved!")
        break

    # Optional: Plot the deformed structure at each iteration
    system_stm.plot_deformed_stm(100, scale=10, title=f'Iteration: {iteration}')
    
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
system_stm.plot_internal_forces_stm()
