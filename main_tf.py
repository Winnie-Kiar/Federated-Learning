import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import collections

# Load MNIST dataset and preprocess
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize

# Convert data to federated format (simulating multiple clients)
NUM_CLIENTS = 5
def preprocess(dataset):
    return dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y)).batch(32)

client_datasets = [tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shard(NUM_CLIENTS, i)
                   for i in range(NUM_CLIENTS)]

# Define Keras model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Wrap model for TFF
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=client_datasets[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Define Federated Averaging process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
)

# Initialize FL state
state = iterative_process.initialize()

# Train Federated Model
NUM_ROUNDS = 10
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, client_datasets)
    print(f"Round {round_num + 1}, Metrics={metrics}")

# Save trained model
keras_model = create_keras_model()
keras_model.save("federated_model.h5")

print("Training Complete. Model Saved.")

