import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data loading function
def load_data(image_folder, data_folder, image_shape):
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

        # Load node image (target)
        data_path = os.path.join(data_folder, data_file)
        node_image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        node_image = node_image.astype(np.float32) / 255.0  # Normalize and ensure float32 type
        node_image = np.expand_dims(node_image, axis=-1)  # Add channel dimension
        y_nodes.append(node_image)

    return np.array(X), np.array(y_nodes)

# Parameters
image_folder = 'generated_random_stms_5nodes/images'
data_folder = 'generated_random_stms_5nodes/generated_node_images'
image_shape = (256, 256, 1)

# Load data
X, y_nodes = load_data(image_folder, data_folder, image_shape)

# Custom loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Encoder part of the autoencoder
def build_encoder(input_shape):
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)
    encoder = models.Model(encoder_input, latent, name='encoder')
    return encoder

# Decoder part of the autoencoder
def build_decoder(output_shape):
    latent_input = layers.Input(shape=(128,))
    x = layers.Dense(32 * 32 * 128, activation='relu')(latent_input)
    x = layers.Reshape((32, 32, 128))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x) # 64x64
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x) # 128x128
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x) # 256x256
    x = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x) # 256x256x1
    decoder = models.Model(latent_input, x, name='decoder')
    return decoder


def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Split into training and test sets
X_train, X_test, y_nodes_train, y_nodes_test = train_test_split(
    X, y_nodes, test_size=0.2, random_state=42)

# Building the autoencoder model
input_shape = (256, 256, 1)

encoder = build_encoder(input_shape)
decoder = build_decoder(input_shape)

autoencoder_input = layers.Input(shape=input_shape)
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')

model = autoencoder

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])

# Summary of the model
autoencoder.summary()

# Train the model and keep track of the training process
batch_size = 256
epochs = 1
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
base_filename = 'autoencoder_model'

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

# Function to create an image from node coordinates
def create_image_from_nodes(nodes, image_shape):
    image = np.ones(image_shape, dtype=np.float32)
    for i in range(0, len(nodes), 2):
        x, y = int(nodes[i] * image_shape[1]), int(nodes[i+1] * image_shape[0])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            image[y, x] = 0.0  # Set node position to black
    return image

# Plot some predictions
def plot_predictions(X, y_nodes_true, y_nodes_pred, num_samples=5):
    for i in range(num_samples):
        true_image = y_nodes_true[i].squeeze()
        predicted_image = y_nodes_pred[i].squeeze()

        plt.figure(figsize=(18, 6))
        
        # Plot the input image
        plt.subplot(1, 3, 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f'Sample {i + 1} Input Image')
        plt.axis('off')

        # Plot the true node image
        plt.subplot(1, 3, 2)
        plt.imshow(true_image, cmap='gray')
        plt.title(f'Sample {i + 1} True Nodes')
        plt.axis('off')

        # Plot the predicted node image
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_image, cmap='gray')
        plt.title(f'Sample {i + 1} Predicted Nodes')
        plt.axis('off')

        plt.show()

# Make predictions
y_nodes_pred = model.predict(X_test)

# Plot some predictions
plot_predictions(X_test, y_nodes_test, y_nodes_pred, num_samples=20)
