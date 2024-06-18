import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Define the CNN + RNN architecture
def create_cnn_rnn(input_shape, max_nodes, max_edges):
    # CNN for feature extraction
    cnn_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    cnn_output = layers.Dense(128, activation='relu')(x)

    # Repeat the feature vector to match the maximum number of nodes and edges
    repeated_features_nodes = layers.RepeatVector(max_nodes)(cnn_output)
    repeated_features_edges = layers.RepeatVector(max_edges)(cnn_output)

    # RNN for node prediction
    rnn_output_nodes = layers.LSTM(128, return_sequences=True)(repeated_features_nodes)
    rnn_output_nodes = layers.LSTM(64, return_sequences=True)(rnn_output_nodes)
    nodes_output = layers.TimeDistributed(layers.Dense(2), name='nodes_output')(rnn_output_nodes)

    # RNN for edge prediction
    rnn_output_edges = layers.LSTM(128, return_sequences=True)(repeated_features_edges)
    rnn_output_edges = layers.LSTM(64, return_sequences=True)(rnn_output_edges)
    edges_output = layers.TimeDistributed(layers.Dense(2), name='edges_output')(rnn_output_edges)
    
    # Create the model
    model = models.Model(inputs=cnn_input, outputs=[nodes_output, edges_output])
    
    return model

# Define input shape (height, width, channels)
input_shape = (256, 256, 1)  # Example input shape, adjust based on your data
max_nodes = 10  # Define the maximum number of nodes per image
max_edges = 15  # Define the maximum number of edges per image

# Create the CNN + RNN model
cnn_rnn_model = create_cnn_rnn(input_shape, max_nodes, max_edges)

# Compile the model
cnn_rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model
cnn_rnn_model.summary()

# Data loading function (from previous example)
def load_data(image_folder, data_folder, max_nodes, max_edges):
    X = []
    y_nodes = []
    y_edges = []

    image_files = sorted(os.listdir(image_folder))
    data_files = sorted(os.listdir(data_folder))

    for image_file, data_file in zip(image_files, data_files):
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0  # Normalize image
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        X.append(image)

        # Load data
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        nodes = []
        edges = []
        reading_nodes = True
        for line in lines:
            if line.strip() == '':
                reading_nodes = False
                continue
            if reading_nodes:
                node = list(map(float, line.strip().split()))
                nodes.append(node)
            else:
                edge = list(map(int, line.strip().split()))
                edges.append(edge)
        
        # Pad nodes and edges to max_nodes and max_edges
        while len(nodes) < max_nodes:
            nodes.append([0.0, 0.0])
        while len(edges) < max_edges:
            edges.append([0, 0])
        
        y_nodes.append(nodes[:max_nodes])
        y_edges.append(edges[:max_edges])
    
    return np.array(X), np.array(y_nodes), np.array(y_edges)

# Parameters
image_folder = 'generated_random_stms/images'
data_folder = 'generated_random_stms/data'
max_nodes = 10
max_edges = 15

# Load data
X, y_nodes, y_edges = load_data(image_folder, data_folder, max_nodes, max_edges)

# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test, y_edges_train, y_edges_test = train_test_split(
    X, y_nodes, y_edges, test_size=0.2, random_state=42)

# Train the model and keep track of the training process
batch_size=32
epochs = 20
history = cnn_rnn_model.fit(X_train, [y_nodes_train, y_edges_train], epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
evaluation = cnn_rnn_model.evaluate(X_test, [y_nodes_test, y_edges_test])

# Print evaluation results
print("Evaluation on test data:")
print(f"Loss: {evaluation[0]}")
print(f"Node Prediction MAE: {evaluation[1]}")
print(f"Edge Prediction MAE: {evaluation[2]}")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['nodes_output_mae'], label='train_nodes_mae')
plt.plot(history.history['val_nodes_output_mae'], label='val_nodes_mae')
plt.plot(history.history['edges_output_mae'], label='train_edges_mae')
plt.plot(history.history['val_edges_output_mae'], label='val_edges_mae')
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.show()

#%% visualize some of the predictions

# Plot some predictions
def plot_predictions(X, y_nodes_true, y_edges_true, y_nodes_pred, y_edges_pred, num_samples=5):
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
            plt.plot(node[0], node[1], 'ro')
        plt.title(f'Sample {i + 1} True Nodes')
        plt.axis('off')

        # Plot the predicted node coordinates
        plt.subplot(1, 3, 3)
        plt.imshow(X[i].squeeze(), cmap='gray')
        for node in nodes_pred:
            plt.plot(node[0], node[1], 'bo')
        plt.title(f'Sample {i + 1} Predicted Nodes')
        plt.axis('off')

        plt.show()

# Make predictions
y_nodes_pred, y_edges_pred = cnn_rnn_model.predict(X_test)

# Plot some predictions
plot_predictions(X_test, y_nodes_test, y_edges_test, y_nodes_pred, y_edges_pred, num_samples=5)
