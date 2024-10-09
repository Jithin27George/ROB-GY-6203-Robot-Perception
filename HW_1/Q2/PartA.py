import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load Fashion-MNIST dataset (10,000 test images)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)  # Flatten the 28x28 images to vectors of size 784
x_test = x_test.reshape(-1, 28 * 28)

# Define Autoencoder in TensorFlow/Keras
latent_dim = 32  # Latent space size

…plt.title("t-SNE visualization of 10,000 Fashion-MNIST images in latent space")
plt.show()255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)  # Flatten the 28x28 images to vectors of size 784
x_test = x_test.reshape(-1, 28 * 28)

# Define Autoencoder in TensorFlow/Keras
latent_dim = 32  # Latent space size

…plt.title("t-SNE visualization of 10,000 Fashion-MNIST images in latent space")
plt.show()255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)  # Flatten the 28x28 images to vectors of size 784
x_test = x_test.reshape(-1, 28 * 28)

# Define Autoencoder in TensorFlow/Keras
latent_dim = 32  # Latent space size

def build_autoencoder():
    # Encoder
    input_img = layers.Input(shape=(28 * 28,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(28 * 28, activation='sigmoid')(decoded)

    # Autoencoder Model
    autoencoder = models.Model(input_img, decoded)

    # Encoder Model (for extracting latent space)
    encoder = models.Model(input_img, encoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Build and train the autoencoder
autoencoder, encoder = build_autoencoder()
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Extract latent representations of the test set
latent_space = encoder.predict(x_test)

# Use t-SNE to reduce the latent space to 2D for visualization
tsne = TSNE(n_components=2, verbose=1)
latent_2d = tsne.fit_transform(latent_space)

# Plot the 2D representation of all 10,000 test images
plt.figure(figsize=(10, 10))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_test, cmap='tab10', s=100)  # 's=2' makes the dots smaller
plt.colorbar()
plt.title("t-SNE visualization of 10,000 Fashion-MNIST images in latent space")
plt.show()
