import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)  # Suppress TensorFlow warnings and errors

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from utils import normalize_nodes, create_cnn_rnn, create_cnn_rnn_with_regularization, custom_loss, create_vgg16_rnn, get_unique_filename, rescale_nodes, plot_predictions, evaluate_and_plot, DataGenerator

# Parameters
image_folder = 'C:/Users/luziu/Desktop/MA/MA Code/generated_random_stms_5nodes_128/images'
data_folder = 'C:/Users/luziu/Desktop/MA/MA Code/generated_random_stms_5nodes_128/data'
max_nodes = 5
image_shape = (128, 128, 1)
num_samples = None  # specify when you don't want to train on the whole dataset. None to use all
m = 'cnn'  # or 'vgg16'
batch_size = 256
epochs = 10

# Create data generators
train_generator = DataGenerator(image_folder, data_folder, max_nodes, image_shape, batch_size=batch_size, num_samples=num_samples)
test_generator = DataGenerator(image_folder, data_folder, max_nodes, image_shape, batch_size=batch_size, num_samples=num_samples)

# Create model
if m == 'cnn':
    model = create_cnn_rnn((128, 128, 1), max_nodes)
elif m == 'vgg16':
    model = create_vgg16_rnn((128, 128, 3), max_nodes)
elif m == 'cnn_regularized':
    model = create_cnn_rnn_with_regularization((128, 128, 1), max_nodes, 0.001)
else:
    print('unknown model')

# Define your initial learning rate
initial_learning_rate = 0.001

# Define your decay schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Use the schedule in your optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model 
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
# model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

# Summary of the model
model.summary()

# Define the checkpoint directory and file format
checkpoint_dir = 'model_checkpoints'
checkpoint_filename = 'cnn_rnn_epoch_{epoch:02d}.h5'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

# Create the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',   # Save every epoch
    save_weights_only=False,  # Save the whole model (not just weights)
    verbose=1            # Print a message when saving the model
)




# Check if GPU is available and set device accordingly
if tf.test.is_gpu_available():
    device_name = '/device:GPU:0'
else:
    device_name = '/device:CPU:0'

# Train the model and pass the checkpoint_callback in the callbacks list
with tf.device(device_name):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[checkpoint_callback]
    )


# Define the base directory and base filename
base_dir = 'models'
base_filename = f"{m}_rnn_{len(train_generator.indices)}_{batch_size}"

# Get a unique filename
unique_filename = get_unique_filename(base_dir, base_filename)

# Save the entire model
model.save(unique_filename)

print(f"Model saved to {unique_filename}")

# Evaluate, plot, and make predictions using the utility function
y_nodes_pred_rescaled = evaluate_and_plot(model, test_generator, history, image_shape, num_samples=20)