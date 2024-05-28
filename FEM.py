import numpy as np
from system import System
from mesh import Mesh

# defining the probelm and solving FE for initial configuration

# geometrie is defined in mesh_test and called by mesh.create()
node_list, element_list  = Mesh.create()
print('number of elements:', len(element_list))

# volume fraction for all elements is set to 1
x = np.ones(len(element_list),dtype=float)

# setting up the system
s = System(node_list, element_list, x, penalty=3)



# mesh 1:
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
# Entweder Zugstab
#s.load_point([4,0],[0.1,0])
# Oder Kragarm unter Biegung
#s.load_point([60,0],[0,-0.1])
s.load_line(np.array([60,0.0]), np.array([60,3.0]),forces=np.array([0.1,0]))
s.apply_dirichlet_bc()

# # mesh 2
# s.fix_line(np.array([0.0,0.0]), np.array([0.0,3.0]))
# s.load_line(np.array([6.0,0.0]), np.array([6.0,3.0]),forces=np.array([0.0,-0.0001]))
# s.apply_dirichlet_bc()

# # mesh 3
# s.fix_node_by_coord(np.array([0.0,-1.0]),[True,True])
# s.fix_node_by_coord(np.array([6.0,-1.0]),[False,True])

# s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
# s.load_point([6,-0.4],[-0.001,0])
# s.load_point([6,0.4],[-0.001,0])
# s.apply_dirichlet_bc()



# solve for initial x vector
u = s.solve_FE()
obj = s.compliance()
dc = s.sensitivity_compliance()
s.plot(deformed=False)
s.plot(deformed=True)


