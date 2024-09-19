import numpy as np
import matplotlib.pyplot as plt
from system import System
from mesh import Mesh

# defining the probelm and solving FE for initial configuration

# geometrie is defined in mesh_test and called by mesh.create()
regular_mesh = False
node_list, element_list  = Mesh.create(regular_mesh)
print('number of elements:', len(element_list))

# volume fraction for all elements is set to 1
x = np.ones(len(element_list),dtype=float)

# setting up the system
s = System(node_list, element_list, x, penalty=3)

for e in element_list:
    e.E = 30000
    e.nu = 0.3
r_min = 0.25 # mesh one
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
#s.fix_node_by_coord([0,-1])
#s.fix_node_by_coord([4,-1])
if regular_mesh == True:
    s.load_point([80,20],[0,-0.1])
else:
    s.load_point([4,-1],[0,-1])
    
s.apply_dirichlet_bc()


# solve for initial x vector
u = s.solve_FE()
obj = s.compliance()
dc = s.sensitivity_compliance()
s.plot(deformed=False)
s.plot(deformed=True)


#%% prinicipal forces

plt.figure()
ax = plt.gca()

for e in element_list:
    sigma_1, sigma_2, alpha = e.principal_stresses_at_element_center()
    sigma_1_vector = sigma_1 * np.array([np.cos(alpha), np.sin(alpha)])
    sigma_2_vector = sigma_2 * np.array([-np.sin(alpha), np.cos(alpha)])
    center = e.element_center()
    
    # Plot sigma_1 as an arrow (principal stress direction)
    ax.quiver(center[0], center[1], sigma_1_vector[0], sigma_1_vector[1], 
              color='r', angles='xy', scale_units='xy', scale=10, label="Sigma_1" if e == element_list[0] else "")
    
    # Plot sigma_2 as an arrow (principal stress direction)
    ax.quiver(center[0], center[1], sigma_2_vector[0], sigma_2_vector[1], 
              color='b', angles='xy', scale_units='xy', scale=10, label="Sigma_2" if e == element_list[0] else "")

# Set plot details
ax.set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Principal Stresses at Element Centers')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
   

