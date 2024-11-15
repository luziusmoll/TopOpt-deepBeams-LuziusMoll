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
# u = s.solve_FE()
# u = s.solve_FE_taichi()
# u = s.solve_FE_sparse()
# Measure the performance of solve_FE_sparse
# start_time = time.time()
# u = s.solve_FE_taichi()
# end_time = time.time()
# print(f"solve_FE_sparse time: {end_time - start_time:.6f} seconds")

# obj = s.compliance()
# dc = s.sensitivity_compliance()
# s.plot(deformed=False)
# s.plot(deformed=True)

#%% comparison of FEM solvers
if 2<4:
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
    

#%% prinicipal forces

# plt.figure()
# ax = plt.gca()

# for e in element_list:
#     sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
#     sigma_1_vector = sigma_1 * np.array([np.cos(alpha), np.sin(alpha)])
#     sigma_2_vector = sigma_2 * np.array([-np.sin(alpha), np.cos(alpha)])
#     center = e.element_center()
    
#     # Plot sigma_1 as an arrow (principal stress direction)
#     ax.quiver(center[0], center[1], sigma_1_vector[0], sigma_1_vector[1], 
#               color='r', angles='xy', scale_units='xy', scale=10, label="Sigma_1" if e == element_list[0] else "")
    
#     # Plot sigma_2 as an arrow (principal stress direction)
#     ax.quiver(center[0], center[1], sigma_2_vector[0], sigma_2_vector[1], 
#               color='b', angles='xy', scale_units='xy', scale=10, label="Sigma_2" if e == element_list[0] else "")

# # Set plot details
# ax.set_aspect('equal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Principal Stresses at Element Centers')

# # Add legend
# plt.legend()

# # Show the plot
# plt.grid(True)
# plt.show()
   

