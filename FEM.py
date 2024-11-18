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
    

#%% test of system with beam elements
from node import Node
from beam_element import BeamElement

  
node1 = Node([0,1], 0, [0,1,2], fixed=[True, True, False], forces=[0,0,0])
node2 = Node([0,-1], 1, [3,4,5], fixed=[True, True, False], forces=[0,0,0])
node3 = Node([2,0], 2, [6,7,8], fixed=[False, False, False], forces=[0,-10,0])
    
node_list = [node1, node2, node3]
 
for node in node_list:
    node.displacements = np.zeros(3)
  

beam1 = BeamElement(0, [node1,node3])
beam2 = BeamElement(1, [node2,node3])


element_list = [beam1, beam2]

        
x = np.ones((len(element_list))) 


system_stm = System(node_list, element_list, x)

# apply dirichlet BCs
system_stm.apply_dirichlet_bc()


u = system_stm.solve_FE()

# plot the stm and its displacements
system_stm.plot_deformation_stm(scale=100)


system_stm.plot_internal_forces_stm()


system_stm.ErgebnissePlotten(100, scale=100)

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
system_canti.ErgebnissePlotten(100, scale=10)