import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
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



def create_cnn_rnn(input_shape, max_nodes):
    # CNN for feature extraction
    cnn_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    cnn_output = layers.Dense(128, activation='relu')(x)
    cnn_output = layers.Dropout(0.5)(cnn_output)

    # Repeat the feature vector to match the maximum number of nodes
    repeated_features_nodes = layers.RepeatVector(max_nodes)(cnn_output)

    # RNN for node prediction
    rnn_output_nodes = layers.LSTM(128, return_sequences=True)(repeated_features_nodes)
    rnn_output_nodes = layers.LSTM(64, return_sequences=True)(rnn_output_nodes)
    nodes_output = layers.TimeDistributed(layers.Dense(2, activation='sigmoid'), name='nodes_output')(rnn_output_nodes)
    
    # Create the model
    model = models.Model(inputs=cnn_input, outputs=nodes_output)
    
    return model

# # using the pretrained vgg16 for feature extraction
# def create_vgg16_rnn(input_shape, max_nodes):
#     # Load pretrained VGG16 model + higher level layers
#     vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#     for layer in vgg16.layers:
#         layer.trainable = False  # Freeze the VGG16 layers

#     # Create the feature extraction model
#     cnn_input = vgg16.input
#     x = vgg16.output
#     x = layers.Flatten()(x)
#     cnn_output = layers.Dense(128, activation='relu')(x)
#     cnn_output = layers.Dropout(0.5)(cnn_output)

#     # Repeat the feature vector to match the maximum number of nodes
#     repeated_features_nodes = layers.RepeatVector(max_nodes)(cnn_output)

#     # RNN for node prediction
#     rnn_output_nodes = layers.LSTM(128, return_sequences=True)(repeated_features_nodes)
#     rnn_output_nodes = layers.LSTM(64, return_sequences=True)(rnn_output_nodes)
#     nodes_output = layers.TimeDistributed(layers.Dense(2, activation='sigmoid'), name='nodes_output')(rnn_output_nodes)
    
#     # Create the model
#     model = models.Model(inputs=cnn_input, outputs=nodes_output)
    
#     return model



# Custom loss function with a Gaussian filter around the correct coordinates
def custom_loss(y_true, y_pred):
    def compute_loss_for_sample(y_true_sample, y_pred_sample):
        # Create a mask to ignore padding values (where y_true is [0, 0])
        mask = tf.reduce_any(y_true_sample != [0.0, 0.0], axis=-1)

        # Filter out the padded nodes
        y_true_sample_filtered = tf.boolean_mask(y_true_sample, mask)
        y_pred_sample_filtered = tf.boolean_mask(y_pred_sample, mask)

        # Ensure y_true_sample_filtered and y_pred_sample_filtered have the same shape
        min_length = tf.minimum(tf.shape(y_true_sample_filtered)[0], tf.shape(y_pred_sample_filtered)[0])
        y_true_sample_filtered = y_true_sample_filtered[:min_length]
        y_pred_sample_filtered = y_pred_sample_filtered[:min_length]
        
        # Calculate the distances between corresponding true and predicted nodes
        distances = tf.sqrt(tf.reduce_sum(tf.square(y_true_sample_filtered - y_pred_sample_filtered), axis=-1))
        
        # Define a Gaussian filter: closer points will have lower loss
        sigma = 0.05
        gaussian_filter = tf.exp(-tf.square(distances) / (2 * tf.square(sigma)))
        
        # Calculate the loss with the Gaussian filter applied
        loss = 1.0 - gaussian_filter
        
        return tf.reduce_mean(loss)
    
    # Apply the loss calculation to each sample in the batch
    batch_losses = tf.map_fn(lambda x: compute_loss_for_sample(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
    
    return tf.reduce_mean(batch_losses)


# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test = train_test_split(
    X, y_nodes, test_size=0.2, random_state=42)

# Define input shape (height, width, channels)
input_shape = (256, 256, 1)
image_shape = (256, 256, 3)

# Create the CNN + RNN model
cnn_rnn_model = create_cnn_rnn(input_shape, max_nodes)
# Create the VGG16 + RNN model
#vgg16_rnn_model = create_vgg16_rnn(image_shape, max_nodes)

model = cnn_rnn_model

# Compile the model with the custom loss function
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
#model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

# Summary of the model
model.summary()

# Train the model and keep track of the training process
batch_size = 256
epochs = 100
history = model.fit(X_train, y_nodes_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)



# Function to get a unique filename
def get_unique_filename(base_dir, base_filename):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    filename = os.path.join(base_dir, base_filename)
    if not os.path.exists(filename + '.h5'):
        return filename + '.h5'
    
    counter = 1
    while True:
        new_filename = f"{filename}_{counter}.h5"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

# Define the base directory and base filename
base_dir = 'models'
base_filename = 'cnn_rnn_model'

# Get a unique filename
unique_filename = get_unique_filename(base_dir, base_filename)

# Save the entire model
model.save(unique_filename)

print(f"Model saved to {unique_filename}")




# Evaluate the model
evaluation = model.evaluate(X_test, y_nodes_test)

# Print evaluation results
print("Evaluation on test data:")
print(f"Loss: {evaluation[0]}")
print(f"Node Prediction MAE: {evaluation[1]}")

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
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.show()


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




