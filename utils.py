import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import Sequence
from tensorflow.keras import regularizers

class DataGenerator(Sequence):
    def __init__(self, image_folder, data_folder, max_nodes, image_shape, batch_size=32, num_samples=None):
        self.image_folder = image_folder
        self.data_folder = data_folder
        self.max_nodes = max_nodes
        self.image_shape = image_shape
        self.batch_size = batch_size
        
        self.image_files = sorted(os.listdir(image_folder))
        self.data_files = sorted(os.listdir(data_folder))
        
        if num_samples is not None:
            self.image_files = self.image_files[:num_samples]
            self.data_files = self.data_files[:num_samples]
        
        self.indices = np.arange(len(self.image_files))
    
    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, y_nodes = self.__data_generation(batch_indices)
        return X, y_nodes
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_indices):
        X = []
        y_nodes = []
        
        for i in batch_indices:
            image_path = os.path.join(self.image_folder, self.image_files[i])
            data_path = os.path.join(self.data_folder, self.data_files[i])
            
            # Load and process image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=-1)
            X.append(image)
            
            # Load and process data
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
            
            nodes = sorted(nodes, key=lambda x: (x[1], x[0]))
            nodes = normalize_nodes(nodes, self.image_shape)
            while len(nodes) < self.max_nodes:
                nodes.append([0.0, 0.0])
            y_nodes.append(nodes[:self.max_nodes])
        
        return np.array(X), np.array(y_nodes)


# Normalize node coordinates to be between 0 and 1
def normalize_nodes(nodes, image_shape):
    return [[node[0] / image_shape[1], node[1] / image_shape[0]] for node in nodes]

# Data loading function (with sorting and normalization)
def load_data(image_folder, data_folder, max_nodes, image_shape, num_samples=None):
    X = []
    y_nodes = []

    image_files = sorted(os.listdir(image_folder))
    data_files = sorted(os.listdir(data_folder))
    
    if num_samples is not None:
        image_files = image_files[:num_samples]
        data_files = data_files[:num_samples]
    
    counter = 0
    for image_file, data_file in zip(image_files, data_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)
        X.append(image)

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
        
        nodes = sorted(nodes, key=lambda x: (x[1], x[0]))
        nodes = normalize_nodes(nodes, image_shape)
        while len(nodes) < max_nodes:
            nodes.append([0.0, 0.0])
        y_nodes.append(nodes[:max_nodes])
        
        if counter%10000==0:
            print('loading data number', counter) 
        counter += 1
    return np.array(X), np.array(y_nodes)


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

def create_cnn_rnn_with_regularization(input_shape, max_nodes, l2_strength=0.001):
    # CNN for feature extraction
    cnn_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', 
                      kernel_regularizer=regularizers.l2(l2_strength))(cnn_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', 
                      kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', 
                      kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    cnn_output = layers.Dense(128, activation='relu', 
                              kernel_regularizer=regularizers.l2(l2_strength))(x)
    cnn_output = layers.Dropout(0.5)(cnn_output)

    # Repeat the feature vector to match the maximum number of nodes
    repeated_features_nodes = layers.RepeatVector(max_nodes)(cnn_output)

    # RNN for node prediction
    rnn_output_nodes = layers.LSTM(128, return_sequences=True, 
                                   kernel_regularizer=regularizers.l2(l2_strength))(repeated_features_nodes)
    rnn_output_nodes = layers.LSTM(64, return_sequences=True, 
                                   kernel_regularizer=regularizers.l2(l2_strength))(rnn_output_nodes)
    nodes_output = layers.TimeDistributed(layers.Dense(2, activation='sigmoid'), name='nodes_output')(rnn_output_nodes)

    # Create the model
    model = models.Model(inputs=cnn_input, outputs=nodes_output)

    return model


# using the pretrained vgg16 for feature extraction
def create_vgg16_rnn(input_shape, max_nodes):
    # Load pretrained VGG16 model + higher level layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg16.layers:
        layer.trainable = False  # Freeze the VGG16 layers

    # Create the feature extraction model
    cnn_input = vgg16.input
    x = vgg16.output
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


def evaluate_and_plot(model, test_generator, history, image_shape, num_samples=5):
    """
    Evaluate the model, plot the results, and make predictions.
    
    Parameters:
    - model: The trained model.
    - test_generator: The data generator for the test set.
    - history: Training history of the model.
    - image_shape: Shape of the input images.
    - num_samples: Number of samples to plot predictions for.
    
    Returns:
    - y_nodes_pred_rescaled: Rescaled predicted node coordinates.
    """
    
    # Check if GPU is available and set device accordingly
    if tf.test.is_gpu_available():
        device_name = '/device:GPU:0'
    else:
        device_name = '/device:CPU:0'
    
    # Evaluate the model
    with tf.device(device_name):
        evaluation = model.evaluate(test_generator)


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

    # Make predictions
    y_nodes_pred = model.predict(test_generator)

    # Get the original images and true nodes from the generator
    X_test, y_nodes_test = test_generator.__getitem__(0)

    # Rescale predicted nodes to original dimensions
    y_nodes_pred_rescaled = np.array([rescale_nodes(pred, image_shape) for pred in y_nodes_pred])

    # Plot some predictions
    plot_predictions(X_test, y_nodes_test, y_nodes_pred, num_samples)

    return y_nodes_pred_rescaled
