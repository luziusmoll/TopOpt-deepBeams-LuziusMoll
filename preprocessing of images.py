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
image_folder = 'datasets/images'
svd_folder = 'datasets/svd'
preprocess_folder = 'datasets/preprocessed'

# Perform SVD on images with binarization
perform_svd_on_images(image_folder, svd_folder, threshold=128)

# Display a few reconstructed images with the first k components
k = 20  # Number of components to use for reconstruction
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
