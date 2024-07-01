import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from utils import load_data, normalize_nodes, rescale_nodes, plot_predictions

# Parameters
image_folder = 'C:/Users/luziu/Desktop/MA/MA Code/generated_random_stms_5nodes/images'
data_folder = 'C:/Users/luziu/Desktop/MA/MA Code/generated_random_stms_5nodes/data'
max_nodes = 5
image_shape = (256, 256, 1)  # Ensure this matches the expected shape for VGG16 or cnn_rnn

# Load data
X, y_nodes = load_data(image_folder, data_folder, max_nodes, image_shape,100)


# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test = train_test_split(
    X, y_nodes, test_size=0.8, random_state=42)

# Define input shape (height, width, channels)
image_shape = (256, 256, 3)


# Load the saved model
model_path = 'C:/Users/luziu/Desktop/MA/MA Code/models/cnn_rnn_epoch_03.h5'  # Adjust the path to your saved model

model = tf.keras.models.load_model(model_path, compile=False)

# Compile the model with a new optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model 
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# # Evaluate the model
# evaluation = model.evaluate(X_test, y_nodes_test)


# Make predictions
y_nodes_pred = model.predict(X_test)

# Rescale predicted nodes to original dimensions
y_nodes_pred_rescaled = np.array([rescale_nodes(pred, image_shape) for pred in y_nodes_pred])

# Plot some predictions
plot_predictions(X_test, y_nodes_test, y_nodes_pred, num_samples=80)



