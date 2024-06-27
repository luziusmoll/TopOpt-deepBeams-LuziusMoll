import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.spatial import Delaunay

def calculate_angle(edge1, edge2):
    vector1 = edge1[1] - edge1[0]
    vector2 = edge2[1] - edge2[0]
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_theta = dot_product / norm_product
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical issues
    return np.degrees(angle)

def is_valid_angle(edges, nodes, min_angle=20):
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            node_indices1 = edges[i]
            node_indices2 = edges[j]
            common_node = set(node_indices1).intersection(set(node_indices2))
            if common_node:
                common_node = list(common_node)[0]
                unique_nodes1 = [n for n in node_indices1 if n != common_node]
                unique_nodes2 = [n for n in node_indices2 if n != common_node]
                if unique_nodes1 and unique_nodes2:
                    edge1 = np.array([nodes[common_node], nodes[unique_nodes1[0]]])
                    edge2 = np.array([nodes[common_node], nodes[unique_nodes2[0]]])
                    angle = calculate_angle(edge1, edge2)
                    if angle < min_angle:
                        return False
    return True

def generate_random_stm(num_nodes, image_size, min_angle=20):
    while True:
        nodes = np.random.rand(num_nodes, 2) * image_size
        tri = Delaunay(nodes)
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted([simplex[0], simplex[1]])))
            edges.add(tuple(sorted([simplex[1], simplex[2]])))
            edges.add(tuple(sorted([simplex[0], simplex[2]])))
        edges = list(edges)
        if is_valid_angle(edges, nodes, min_angle):
            break
    return nodes, edges

def draw_stm(nodes, edges, image_size, truss_width=6):
    fig, ax = plt.subplots()
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    for edge in edges:
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 'k-', lw=truss_width)
    for node in nodes:
        ax.plot(node[0], node[1], 'ro')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_noise(image, sigma=25):
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_erosion(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=3)
    return eroded_image

def generate_and_save_samples(num_samples, image_size, intermediate_resolution, final_resolution, output_folder, plot_samples=10):
    image_folder = os.path.join(output_folder, 'images')
    data_folder = os.path.join(output_folder, 'data')
    
    # Create directories if they don't exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    for i in range(num_samples):
        #print(f"Generating sample {i + 1}/{num_samples}")
        num_nodes = 5
        # num_nodes = np.random.randint(3, 7)
        nodes, edges = generate_random_stm(num_nodes, image_size)
        image = draw_stm(nodes, edges, image_size)
        
        # Convert to grayscale
        gray_image = convert_to_grayscale(image)
        
        # Apply noise
        noisy_image = apply_noise(gray_image)
        
        # Resize image to intermediate resolution and back to final resolution
        noisy_image_resized = cv2.resize(noisy_image, (intermediate_resolution, intermediate_resolution), interpolation=cv2.INTER_AREA)
        noisy_image_resized = cv2.resize(noisy_image_resized, (final_resolution, final_resolution), interpolation=cv2.INTER_LINEAR)
        
        # Apply erosion
        final_image = apply_erosion(noisy_image_resized)
        
        # Save the final image
        image_path = os.path.join(image_folder, f'image_{i + 1}.png')
        cv2.imwrite(image_path, final_image)
        
        # Verify if the image was saved correctly
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} was not saved correctly.")
        
        # Flip the y-coordinates of the nodes
        flipped_nodes = nodes.copy()
        flipped_nodes[:, 1] = image_size - flipped_nodes[:, 1]

        # Save the node coordinates and edges to a text file
        data_path = os.path.join(data_folder, f'data_{i + 1}.txt')
        if i == 0:
            print(f"Saving data to image = apply_svd(image, 50)  # Apply SVD{data_path}")
        with open(data_path, 'w') as file:
            for node in flipped_nodes:
                file.write(f"{node[0]} {node[1]}\n")
            file.write("\n")  # Separate nodes and edges
            for edge in edges:
                file.write(f"{edge[0]} {edge[1]}\n")
        
        # Verify if the data was saved correctly
        if not os.path.exists(data_path):
            print(f"Error: Data {data_path} was not saved correctly.")
        
        # Plot only the first `plot_samples` samples
        if i < plot_samples:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(noisy_image, cmap='gray')
            plt.title(f'Noisy Sample {i + 1} (Original Resolution)')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(final_image, cmap='gray')
            plt.title(f'Final Sample {i + 1} (Final Resolution)')
            plt.axis('off')
            plt.show()
        
        if i %100 == 0:
            print(f'sample {i + 1}/{num_samples}')
        
        # # Plot the saved image along with nodes and edges
        # plt.figure()
        # plt.imshow(final_image, cmap='gray')
        # plt.scatter(flipped_nodes[:, 0], flipped_nodes[:, 1], c='r')
        # for edge in edges:
        #     node1 = flipped_nodes[edge[0]]
        #     node2 = flipped_nodes[edge[1]]
        #     plt.plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
        # plt.title(f'Sample {i + 1} with Nodes and Edges')
        # plt.axis('off')
        # plt.show()

# Parameters
num_samples = 100000
image_size = 256
intermediate_resolution = 64
final_resolution = 256
output_folder = 'generated_random_stms_5nodes'

# Generate and save samples
generate_and_save_samples(num_samples, image_size, intermediate_resolution, final_resolution, output_folder)

#%% output images with only the nodes black

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a binary mask image with node positions
def create_node_image(nodes, image_shape):
    image = np.ones(image_shape, dtype=np.float32)
    for node in nodes:
        x, y = int(node[0]), int(node[1])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:  # Ensure coordinates are within bounds
            image[y, x] = 0.0  # Set node position to black
    return image

# Data loading function to create images with node positions
def load_data_and_create_node_images(image_folder, data_folder, output_folder, image_shape):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(os.listdir(image_folder))
    data_files = sorted(os.listdir(data_folder))
    
    for image_file, data_file in zip(image_files, data_files):
        # Load original image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize image and ensure float32 type

        # Load node data
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        nodes = []
        for line in lines:
            if line.strip():  # Stop reading if an empty line is encountered
                node = list(map(float, line.strip().split()))
                nodes.append(node)
            else:
                break

        # Create binary mask image with node positions
        node_image = create_node_image(nodes, (image_shape[0], image_shape[1]))

        # Save the generated node image
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, (node_image * 255).astype(np.uint8))

    print(f"Node images generated and saved to {output_folder}")
    return image_files

# Parameters
image_folder = 'generated_random_stms_5nodes/images'
data_folder = 'generated_random_stms_5nodes/data'
output_folder = 'generated_random_stms_5nodes/generated_node_images'
image_shape = (256, 256)  # Ensure this matches the expected shape for the output images

# Load data and create node images
image_files = load_data_and_create_node_images(image_folder, data_folder, output_folder, image_shape)

# Verification of the first few images
output_files = sorted(os.listdir(output_folder))
for i in range(min(5, len(output_files))):  # Ensure not to go out of range
    input_image_path = os.path.join(image_folder, image_files[i])
    node_image_path = os.path.join(output_folder, output_files[i])

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    node_image = cv2.imread(node_image_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title(f'Input Image {i+1}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(node_image, cmap='gray')
    plt.title(f'Node Image {i+1}')
    plt.axis('off')

    plt.show()
    
    