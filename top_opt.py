import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from system import System
from mesh import Mesh

# parameters:
volfrac=0.4
penalty = 3
x_min = 1e-3
ft=0        # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
max_iteration = 30 
mesh_ind_filter = True
regular_mesh = False

# r_min needs to be according to  dimensions of the problem
if regular_mesh == True:
    r_min = 3
else:
    r_min = 0.2
    
    
# set up geometry as defined in mesh_test
node_list, element_list  = Mesh.create(regular_mesh)
print('number of elements:', len(element_list))

for e in element_list:
    e.E = 30000
    e.nu = 0.3

# volume fraction for all elements is set to volfrac
x = np.ones(len(element_list),dtype=float)*volfrac

# Set up FE problem
s = System(node_list, element_list, x, penalty, x_min)


# Apply boundary conditions to structure
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
#s.fix_node_by_coord([4,0])
if regular_mesh == True:
    s.load_point([80,20],[0,-1])
else:
    s.load_point([4,0],[0,-1])
#s.load_line(np.array([60,0.0]), np.array([60,3.0]),forces=np.array([0.0,-0.01]))
s.apply_dirichlet_bc()



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
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
        # possibility to define passive areas
        if regular_mesh == True: 
            for ely in range(40):
                for elx in range(80):
                    if np.sqrt((ely-20)**2 + (elx-30)**2) < 10:
                        xnew[elx*40+ely] = x_min
        
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (xnew,gt)


#%% Actual optimization 
""" from DTU's minimum compliance problem (basic 200 lines python code) https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python """

# Set loop counter and gradient vectors 
loop=0
obj_hist = []
change=1
dv = np.ones(len(element_list))
dc = np.ones(len(element_list))
ce = np.ones(len(element_list))
# The following must be initialized to use the NGuyen/Paulino OC approachgls
xold=x.copy()
xPhys=x.copy()
g=0 
while change>0.001 and loop<max_iteration: 
    loop=loop+1
    
    # Solve FE problem
    print(loop)
    u = s.solve_FE() 
    
    #K_g = s.K_global()
    #print(K_g)
    # Objective and sensitivity
    obj=s.compliance()
    obj_hist.append(obj)
    # according to sigmund2001 eq4 (no filter)
    dc=s.sensitivity_compliance()  
    
    # according to sigmund2001 eq5 (with filter)
    if mesh_ind_filter == True:
        dc_filtered = []
        for i in range(len(element_list)):
            dc_filtered_i = 1 / x[i] * np.sum(H_f[:,i]) * np.sum( H_f[:,i] * x * dc)
            dc_filtered.append(dc_filtered_i)
            
        dc= dc_filtered
        
    
    dv = np.ones(len(element_list))
    # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
    # if ft==0:
    #     dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
    # elif ft==1:
    #     dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
    #     dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
    # Optimality criteria
    xold[:]=x
    (x[:],g)=oc(len(element_list),x,volfrac,dc,dv,g)
    # pass new x vector to system
    s.x = x
    # Filter design variables
    # if ft==0:   xPhys[:]=x
    # elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
    # Compute the change by the inf. norm 
    change=np.linalg.norm(x.reshape(len(element_list),1)-xold.reshape(len(element_list),1),np.inf)
    # Write iteration history to screen (req. Python 2.6 or newer)
    print('obj:',obj)
    print('change:', change)
    print('mean x:',np.mean(x))
    #print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    if (loop - 1) % 3 == 0:
        #s.plot2(deformed=False)
        s.plot2(deformed=True)
 

s.plot2(deformed=False)


#%% combined plot for obsidian

def combined_plot(s):
    fig = plt.figure(figsize=(18, 5))  # Overall figure size
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Adjust the middle plot width if needed
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Plotting the optimized structure using plot3 method
    s.plot3(ax=ax1, deformed=False)
    ax1.set_title('Mesh Plot')
    ax1.set_aspect('equal')  # Set to 'equal' to maintain original scale (otherwise 'auto')
    
    # Plotting Objective History
    ax2.plot(obj_hist)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective')
    ax2.set_title('Objective History')
    ax2.grid(True)
    
    # Plotting the distribution of x
    ax3.hist(x, bins=30, alpha=0.75)
    ax3.set_title('Histogram of x')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
combined_plot(s)

