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
    r_min = 4
else:
    r_min = 0.25
    
    
# set up geometry as defined in mesh_test
node_list, element_list  = Mesh.create(regular_mesh)
print('number of elements:', len(element_list))

for e in element_list:
    e.E = 30000
    e.nu = 0.15

# volume fraction for all elements is set to volfrac
x = np.ones(len(element_list),dtype=float)*volfrac

# Set up FE problem
s = System(node_list, element_list, x, penalty, x_min)


# Apply boundary conditions to structure
s.fix_line(np.array([0.0,-1.0]), np.array([0.0,1.0]))
#s.fix_node_by_coord([0,-1])
#s.fix_node_by_coord([4,-1])
if regular_mesh == True:
    s.load_point([80,20],[0,-0.1])
else:
    s.load_point([4,-1],[0,-1])
    #s.load_point([1.5,-1],[0,-1])
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

#%% Second optimization 
# max_iteration=1

# def oc(n_ele,x,volfrac,dc,dv,g):
#     dc=np.array(dc)
#     l1=0
#     l2=1e9
#     move=0.2
#     # reshape to perform vector operations
#     xnew=np.zeros(n_ele)
#     while (l2-l1)/(l1+l2)>1e-3:
#         lmid=0.5*(l2+l1)
#         xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        
#         # possibility to define passive areas
#         if regular_mesh == True: 
#             for ely in range(40):
#                 for elx in range(80):
#                     if np.sqrt((ely-20)**2 + (elx-30)**2) < 10:
#                         xnew[elx*40+ely] = x_min
        
#         for i in range(len(xnew)):
#             if xnew[i] < 0.6:
#                 xnew[i] = 0
#         gt=g+np.sum((dv*(xnew-x)))
#         if gt>0 :
#             l1=lmid
#         else:
#             l2=lmid
#     return (xnew,gt)


# # Set loop counter and gradient vectors 
# loop=0
# obj_hist = []
# change=1
# dv = np.ones(len(element_list))
# dc = np.ones(len(element_list))
# ce = np.ones(len(element_list))
# # The following must be initialized to use the NGuyen/Paulino OC approachgls
# xold=x.copy()
# xPhys=x.copy()
# g=0 
# while change>0.001 and loop<max_iteration: 
#     loop=loop+1
    
#     # Solve FE problem
#     print(loop)
#     u = s.solve_FE() 
    
#     #K_g = s.K_global()
#     #print(K_g)
#     # Objective and sensitivity
#     obj=s.compliance()
#     obj_hist.append(obj)
#     # according to sigmund2001 eq4 (no filter)
#     dc=s.sensitivity_compliance()  
    
#     # according to sigmund2001 eq5 (with filter)
#     if mesh_ind_filter == True:
#         dc_filtered = []
#         for i in range(len(element_list)):
#             dc_filtered_i = 1 / x[i] * np.sum(H_f[:,i]) * np.sum( H_f[:,i] * x * dc)
#             dc_filtered.append(dc_filtered_i)
            
#         dc= dc_filtered
        
    
#     dv = np.ones(len(element_list))
#     # Sensitivity filtering: ft==0 -> sens, ft==1 -> dens
#     # if ft==0:
#     #     dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
#     # elif ft==1:
#     #     dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
#     #     dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
#     # Optimality criteria
#     xold[:]=x
#     (x[:],g)=oc(len(element_list),x,volfrac,dc,dv,g)
#     # pass new x vector to system
#     s.x = x
#     # Filter design variables
#     # if ft==0:   xPhys[:]=x
#     # elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
#     # Compute the change by the inf. norm 
#     change=np.linalg.norm(x.reshape(len(element_list),1)-xold.reshape(len(element_list),1),np.inf)
#     # Write iteration history to screen (req. Python 2.6 or newer)
#     print('obj:',obj)
#     print('change:', change)
#     print('mean x:',np.mean(x))
#     #print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
#     if (loop - 1) % 3 == 0:
#         #s.plot2(deformed=False)
#         s.plot2(deformed=True)

# for x in s.x:
#     if x<0.9:
#         x=0

# s.plot2(deformed=False)
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


#%% processing and saving of image
import cv2
import os
from skimage.morphology import skeletonize
from skimage.util import invert

def save_plot_as_image(plot_variable, folder_name="preprocessed_images"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    existing_files = [f for f in os.listdir(folder_name) if f.endswith('.png')]
    if existing_files:
        numbers = [int(f.split('.')[0].split('_')[-1]) for f in existing_files]
        highest_number = max(numbers)
    else:
        highest_number = 0
    
    new_number = highest_number + 1
    filename = f"topology_plot_{new_number}.png"
    filepath = os.path.join(folder_name, filename)
    plot_variable.savefig(filepath, bbox_inches='tight', pad_inches=0)
    
    print(f"Saved plot as {filepath}")
    return filepath

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape

    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_rescaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]  # White color
    image_padded = cv2.copyMakeBorder(image_rescaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    image_normalized = image_padded / 255.0

    transformation_rule = {
        'scale': scale,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'original_shape': original_shape,
        'target_size': target_size
    }

    # plt.figure(figsize=(6, 6))
    # plt.imshow(image_normalized)
    # plt.axis('off')
    # plt.title("Preprocessed Image")
    # plt.show()

    return image_normalized, transformation_rule

def apply_transformation(coord, transformation_rule):
    scale = transformation_rule['scale']
    top = transformation_rule['top']
    left = transformation_rule['left']
    original_shape = transformation_rule['original_shape']

    x, y = coord
    x_new = int(x * scale) + left
    y_new = int(y * scale) + top

    return (x_new, y_new)

def reverse_transformation(coord, transformation_rule):
    scale = transformation_rule['scale']
    top = transformation_rule['top']
    left = transformation_rule['left']

    x, y = coord
    x_orig = (x - left) / scale
    y_orig = (y - top) / scale

    return (int(x_orig), int(y_orig))

# Generate and save the plot
plot_variable = s.plot4(deformed=False) 
image_path = save_plot_as_image(plot_variable)

# Preprocess the image
target_size = 256  # Example target size

preprocessed_image, transformation_rule = preprocess_image(image_path, target_size)

# Convert to 8-bit unsigned integer and grayscale
preprocessed_image_uint8 = (preprocessed_image * 255).astype(np.uint8)
gray = cv2.cvtColor(preprocessed_image_uint8, cv2.COLOR_RGB2GRAY)



# Apply thresholding to convert to binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


# Apply erosion to thin the structures
kernel_size = 3 # kernel size 1 means not doing anything
kernel = np.ones((kernel_size, kernel_size), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)


# Apply dilation to smooth the structure
kernel_size = 5 # kernel size 1 means not doing anything
kernel = np.ones((kernel_size, kernel_size), np.uint8)
dilated = cv2.dilate(eroded, kernel, iterations=2)

# Apply skeletonization
skeleton = skeletonize(dilated > 0)
skeleton_uint8 = (skeleton * 255).astype(np.uint8)

# Apply Gaussian blur to smooth the edges
kernel_size = 5
smoothed = cv2.GaussianBlur(skeleton_uint8, (kernel_size, kernel_size), 2)

# Plot the results
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(eroded, cmap='gray')
plt.title("Eroded Image")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(skeleton_uint8, cmap='gray')
plt.title("Skeleton")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(smoothed, cmap='gray')
plt.title("Smoothed Skeleton")
plt.axis('off')

plt.tight_layout()
plt.show()



#%% Edge and line detectionand  intersection detections for lines that intersect at e.g. an angle > 20°


# Apply Canny Edge Detection
low_threshold = 10
high_threshold = 200
edges = cv2.Canny(smoothed, low_threshold, high_threshold)

# Hough Transform parameters
rho = 1.4  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments
line_image = np.copy(smoothed) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)



# Function to detect the intersection of two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return int(px), int(py)
    return None

# Function to calculate the angle between two lines
def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    angle = np.abs(angle1 - angle2)
    if angle > np.pi:
        angle = 2 * np.pi - angle
    return angle * 180 / np.pi

# Function to cluster close points
def cluster_points(points, threshold=10):
    if not points:
        return []
    
    clusters = []
    used = [False] * len(points)
    
    for i, point in enumerate(points):
        if not used[i]:
            cluster = [point]
            used[i] = True
            for j, other_point in enumerate(points):
                if not used[j]:
                    dist = np.sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2)
                    if dist < threshold:
                        cluster.append(other_point)
                        used[j] = True
            clusters.append(cluster)
    
    combined_points = []
    for cluster in clusters:
        avg_x = int(np.mean([p[0] for p in cluster]))
        avg_y = int(np.mean([p[1] for p in cluster]))
        combined_points.append((avg_x, avg_y))
    
    return combined_points

# Detect intersections and calculate angles
intersections = []
if lines is not None:
    num_lines = len(lines)
    for i in range(num_lines):
        for j in range(i + 1, num_lines):
            line1 = lines[i][0]
            line2 = lines[j][0]
            intersect = line_intersection(line1, line2)
            if intersect:
                angle = calculate_angle(line1, line2)
                if angle > 20:
                    intersections.append(intersect)

# Cluster close intersection points
combined_intersections = cluster_points(intersections, threshold=20)

# Reinitialize line_image to draw combined intersections
line_image = np.zeros_like(preprocessed_image_uint8)

# Draw the detected lines on the line_image
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

# Draw the combined intersections on the line image
for point in combined_intersections:
    cv2.circle(line_image, point, 5, (0, 255, 0), -1)
# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(smoothed, cmap='gray')
plt.title("Grayscale Image of Skeleton")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(line_image)
plt.title("Detected Lines with Intersections")
plt.axis('off')

plt.tight_layout()
plt.show() 


print("Intersections detected and plotted.")

