import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_images(image_folder):
    """ Load images from the specified folder. """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        if image is not None:
            images.append((image_file, image))
        else:
            print(f"Error: Unable to read image {image_path}")
    return images

def make_binary(image, threshold=128):
    """ Convert image to binary using the given threshold. """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_erosion(image, kernel_size=5, iterations=3):
    """ Apply erosion to the binary image. """
    # Flip black and white
    flipped_image = cv2.bitwise_not(image)
    
    # Smooth image
    smoothed_image = cv2.GaussianBlur(flipped_image, (kernel_size, kernel_size), 0)
    
    # Apply erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(smoothed_image, kernel, iterations=iterations)
    
    # Flip black and white back
    eroded_image = cv2.bitwise_not(eroded_image)
    return eroded_image

def compute_svd(image):
    """ Compute the Singular Value Decomposition (SVD) of a binary image. """
    U, S, V = np.linalg.svd(image, full_matrices=False)
    return U, S, V

def perform_svd_on_images(image_folder, svd_folder, threshold=128):
    """ Read images from the image folder, binarize them, perform SVD, and save the results to the svd folder. """
    # Create the SVD folder if it doesn't exist
    if not os.path.exists(svd_folder):
        os.makedirs(svd_folder)
    
    # Load images
    images = load_images(image_folder)
    
    for i, (image_file, image) in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}: {image_file}")
        
        # Binarize the image
        binary_image = make_binary(image, threshold)
        
        # Compute SVD
        U, S, V = compute_svd(binary_image)
        
        # Save the SVD components
        svd_path = os.path.join(svd_folder, f'svd_{os.path.splitext(image_file)[0]}.npz')
        print(f"Saving SVD to {svd_path}")
        np.savez(svd_path, U=U, S=S, V=V)
        
        # Verify if the SVD was saved correctly
        if not os.path.exists(svd_path):
            print(f"Error: SVD {svd_path} was not saved correctly.")

def reconstruct_image(U, S, V, k=None):
    """ Reconstruct an image from its SVD components using the first k components. """
    if k is None or k > len(S):
        k = len(S)
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:k, :]
    return np.dot(U_k, np.dot(np.diag(S_k), V_k))

def display_reconstructed_images(svd_folder, num_examples=5, k=None, threshold=128):
    """ Load SVD components from files, reconstruct images with the first k components, binarize them, apply erosion, and display them. """
    # List all SVD files in the svd folder
    svd_files = [f for f in os.listdir(svd_folder) if f.endswith('.npz')]
    
    for i, svd_file in enumerate(svd_files[:num_examples]):
        svd_path = os.path.join(svd_folder, svd_file)
        print(f"Reconstructing image from {svd_path}")
        
        # Load the SVD components
        with np.load(svd_path) as data:
            U = data['U']
            S = data['S']
            V = data['V']
        
        # Reconstruct the image using the first k components
        reconstructed_image = reconstruct_image(U, S, V, k)
        
        # Binarize the reconstructed image
        binary_reconstructed_image = make_binary(reconstructed_image, threshold)
        
        # Apply erosion to the binary reconstructed image
        eroded_image = apply_erosion(binary_reconstructed_image)
        
        # Display the eroded binary reconstructed image
        plt.figure(figsize=(6, 6))
        plt.imshow(eroded_image, cmap='gray')
        plt.title(f'Reconstructed Image {i + 1} with k={k}')
        plt.axis('off')
        plt.show()

def save_reconstructed_image(image, preprocess_folder, image_name):
    """ Save the reconstructed image to the specified folder. """
    if not os.path.exists(preprocess_folder):
        os.makedirs(preprocess_folder)
    image_path = os.path.join(preprocess_folder, image_name)
    cv2.imwrite(image_path, image)
    print(f"Saved reconstructed image to {image_path}")

# Parameters
image_folder = 'topopt_result'
svd_folder = 'datasets/svd'
preprocess_folder = 'datasets/preprocessed'

if 2<0:
    # Perform SVD on images with binarization
    perform_svd_on_images(image_folder, svd_folder, threshold=150)
    
    # Display a few reconstructed images with the first k components
    k = 10  # Number of components to use for reconstruction
    display_reconstructed_images(svd_folder, num_examples=6, k=k, threshold=128)
    
    # Save reconstructed images for all k values in a loop
    svd_files = [f for f in os.listdir(svd_folder) if f.endswith('.npz')]
    for svd_file in svd_files:
        svd_path = os.path.join(svd_folder, svd_file)
        with np.load(svd_path) as data:
            U = data['U']
            S = data['S']
            V = data['V']
    
        reconstructed_image = reconstruct_image(U, S, V, k)
        binary_reconstructed_image = make_binary(reconstructed_image, threshold=128)
        eroded_image = apply_erosion(binary_reconstructed_image)
        image_name = f'reconstructed_{os.path.splitext(svd_file)[0]}_k{k}.png'
        save_reconstructed_image(eroded_image, preprocess_folder, image_name)


import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def line_detection_plot(binary,smoothed, skeleton_uint8,smoothed_skel,edges,line_image):
    plt.figure(figsize=(26, 6))
    
    plt.subplot(1, 6, 1)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(smoothed, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(skeleton_uint8, cmap='gray')
    plt.title("Skeleton")
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(smoothed_skel, cmap='gray')
    plt.title("Smoothed Skeleton")
    plt.axis('off')
    
    plt.subplot(1, 6, 5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')
    
    plt.subplot(1, 6, 6)
    plt.imshow(line_image)
    plt.title("Detected Lines and Intersections")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def segmentation(image, min_samples=1, min_line_length=5, max_line_gap=5, showplot=True):
    # Since the image is already grayscale, no need for conversion
    gray_image = image

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Apply Gaussian blur to smooth the edges
    kernel_size = 3
    smoothed = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 2)

    # Apply skeletonization (using thresholding for binary image)
    skeleton = skeletonize(smoothed > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Apply Gaussian blur to smooth the edges
    smoothed_skel = cv2.GaussianBlur(skeleton_uint8, (kernel_size, kernel_size), 2)

    # Edge and line detection using Canny
    low_threshold = 10
    high_threshold = 200
    edges = cv2.Canny(skeleton_uint8, low_threshold, high_threshold)

    # Hough Transform parameters
    rho = 1.5
    theta = 0.1 #np.pi / 180  
    threshold = 10
    line_image = np.zeros_like(image)

    # Detect and draw lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Plot the images
    line_detection_plot(inverted_image, smoothed, skeleton_uint8,smoothed_skel,edges,line_image)
    
    return line_image


image_folder = 'topopt_result'
segmented_folder = 'datasets/segmented'

images = load_images(image_folder)

for image in images:
    segmentation(image[1])













