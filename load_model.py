<<<<<<< Updated upstream
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split


# Normalize node coordinates to be between 0 and 1
def normalize_nodes(nodes, image_shape):
    return [[node[0] / image_shape[1], node[1] / image_shape[0]] for node in nodes]


# Data loading function (with sorting and normalization)
def load_data(image_folder, data_folder, max_nodes, image_shape):
    X = []
    y_nodes = []

    image_files = sorted(os.listdir(image_folder))
    data_files = sorted(os.listdir(data_folder))

    for image_file, data_file in zip(image_files, data_files):
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize image and ensure float32 type
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.repeat(image, 3, axis=-1)  # Repeat channel to match VGG16 input
        X.append(image)

        # Load data
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        nodes = []
        reading_nodes = True
        for line in lines:
            if line.strip() == '':
                reading_nodes = False
                continue
            if reading_nodes:
                node = list(map(float, line.strip().split()))
                nodes.append(node)
        
        # Sort nodes from top-left to bottom-right
        nodes = sorted(nodes, key=lambda x: (x[1], x[0]))  # Sort by y first, then by x

        # Normalize node coordinates
        nodes = normalize_nodes(nodes, image_shape)
        
        # Pad nodes to max_nodes
        while len(nodes) < max_nodes:
            nodes.append([0.0, 0.0])
        
        y_nodes.append(nodes[:max_nodes])
    
    return np.array(X), np.array(y_nodes)

# Parameters
image_folder = 'generated_random_stms_5nodes/images'
data_folder = 'generated_random_stms_5nodes/data'
max_nodes = 5
image_shape = (256, 256, 3)  # Ensure this matches the expected shape for VGG16

# Load data
X, y_nodes = load_data(image_folder, data_folder, max_nodes, image_shape)


# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test = train_test_split(
    X, y_nodes, test_size=0.05, random_state=42)

# Define input shape (height, width, channels)
image_shape = (256, 256, 3)


# Load the saved model
model_path = 'models/vgg16_rnn_model_first_one_that_kind_of_works.h5'  # Adjust the path to your saved model
model = tf.keras.models.load_model(model_path)

# # Evaluate the model
# evaluation = model.evaluate(X_test, y_nodes_test)


# Rescale node coordinates to original dimensions
def rescale_nodes(nodes, image_shape):
    return [[node[0] * image_shape[1], node[1] * image_shape[0]] for node in nodes]

# Plot some predictions
def plot_predictions(X, y_nodes_true, y_nodes_pred, num_samples=5):
    for i in range(num_samples):
        # Get the true and predicted node coordinates
        nodes_true = y_nodes_true[i]
        nodes_pred = y_nodes_pred[i]

        plt.figure(figsize=(18, 6))
        
        # Plot the input image
        plt.subplot(1, 3, 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f'Sample {i + 1} Input Image')
        plt.axis('off')

        # Plot the true node coordinates
        plt.subplot(1, 3, 2)
        plt.imshow(X[i].squeeze(), cmap='gray')
        for node in nodes_true:
            if node[0] == 0 and node[1] == 0:
                continue  # Skip padding nodes
            plt.plot(node[0] * 256, node[1] * 256, 'ro')  # Rescale coordinates
        plt.title(f'Sample {i + 1} True Nodes')
        plt.axis('off')

        # Plot the predicted node coordinates
        plt.subplot(1, 3, 3)
        plt.imshow(X[i].squeeze(), cmap='gray')
        for node in nodes_pred:
            plt.plot(node[0] * 256, node[1] * 256, 'bo')  # Rescale coordinates
        plt.title(f'Sample {i + 1} Predicted Nodes')
        plt.axis('off')

        plt.show()

# Make predictions
y_nodes_pred = model.predict(X_test)

# Rescale predicted nodes to original dimensions
y_nodes_pred_rescaled = np.array([rescale_nodes(pred, image_shape) for pred in y_nodes_pred])

# Plot some predictions
plot_predictions(X_test, y_nodes_test, y_nodes_pred, num_samples=20)


print(y_nodes_test[1]*256)

print(y_nodes_pred[1]*256)




=======
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split


# Normalize node coordinates to be between 0 and 1
def normalize_nodes(nodes, image_shape):
    return [[node[0] / image_shape[1], node[1] / image_shape[0]] for node in nodes]


def load_data(image_folder, data_folder, max_nodes, image_shape, max_files=20):
    X = []
    y_nodes = []

    image_files = sorted(os.listdir(image_folder))[:max_files]
    data_files = sorted(os.listdir(data_folder))[:max_files]

    for image_file, data_file in zip(image_files, data_files):
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize image and ensure float32 type
        image = np.expand_dims(image, axis=-1)  # Add channel dimension to make it (256, 256, 1)
        X.append(image)

        # Load data
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        nodes = []
        reading_nodes = True
        for line in lines:
            if line.strip() == '':
                reading_nodes = False
                continue
            if reading_nodes:
                node = list(map(float, line.strip().split()))
                nodes.append(node)
        
        # Sort nodes from top-left to bottom-right
        nodes = sorted(nodes, key=lambda x: (x[1], x[0]))  # Sort by y first, then by x

        # Normalize node coordinates
        nodes = normalize_nodes(nodes, image_shape)
        
        # Pad nodes to max_nodes
        while len(nodes) < max_nodes:
            nodes.append([0.0, 0.0])
        
        y_nodes.append(nodes[:max_nodes])
    
    return np.array(X), np.array(y_nodes)

# Parameters
image_folder = 'generated_random_stms_5nodes/images'
data_folder = 'generated_random_stms_5nodes/data'
max_nodes = 5
image_shape = (256, 256, 1)  # Ensure this matches the expected shape for the model

# Load data
X, y_nodes = load_data(image_folder, data_folder, max_nodes, image_shape)

# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test = train_test_split(
    X, y_nodes, test_size=0.8, random_state=42)

# Define input shape (height, width, channels)
image_shape = (256, 256, 1)

# Load the saved model with a custom optimizer
model_path = 'models/cnn_rnn_epoch_30.h5'  # Adjust the path to your saved model

# Reload the model with the appropriate optimizer
model = tf.keras.models.load_model(model_path,  compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')

# Rescale node coordinates to original dimensions
def rescale_nodes(nodes, image_shape):
    return [[node[0] * image_shape[1], node[1] * image_shape[0]] for node in nodes]

# Plot some predictions
def plot_predictions(X, y_nodes_true, y_nodes_pred, num_samples=15):
    for i in range(num_samples):
        # Get the true and predicted node coordinates
        nodes_true = y_nodes_true[i]
        nodes_pred = y_nodes_pred[i]

        plt.figure(figsize=(18, 6))
        
        # Plot the input image
        plt.subplot(1, 3, 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f'Sample {i + 1} Input Image')
        plt.axis('off')

        # Plot the true node coordinates
        plt.subplot(1, 3, 2)
        plt.imshow(X[i].squeeze(), cmap='gray')
        for node in nodes_true:
            if node[0] == 0 and node[1] == 0:
                continue  # Skip padding nodes
            plt.plot(node[0] * 256, node[1] * 256, 'ro')  # Rescale coordinates
        plt.title(f'Sample {i + 1} True Nodes')
        plt.axis('off')

        # Plot the predicted node coordinates
        plt.subplot(1, 3, 3)
        plt.imshow(X[i].squeeze(), cmap='gray')
        for node in nodes_pred:
            plt.plot(node[0] * 256, node[1] * 256, 'bo')  # Rescale coordinates
        plt.title(f'Sample {i + 1} Predicted Nodes')
        plt.axis('off')

        plt.show()

# Make predictions
y_nodes_pred = model.predict(X_test)

# Rescale predicted nodes to original dimensions
y_nodes_pred_rescaled = np.array([rescale_nodes(pred, image_shape) for pred in y_nodes_pred])

# Plot some predictions
plot_predictions(X_test, y_nodes_test, y_nodes_pred, num_samples=15)

>>>>>>> Stashed changes
