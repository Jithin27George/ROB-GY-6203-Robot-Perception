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
autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Extract latent representations of the training and test sets
latent_train = encoder.predict(x_train)
latent_test = encoder.predict(x_test)

# Build the classifier
def build_classifier():
    input_latent = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(input_latent)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)
    classifier = models.Model(input_latent, output)
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier

classifier = build_classifier()
classifier.summary()

# Train the classifier
history = classifier.fit(
    latent_train, y_train,
    epochs=20,
    batch_size=256,
    validation_data=(latent_test, y_test)
)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Classifier Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Classifier Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Evaluate the classifier on the test set
test_loss, test_accuracy = classifier.evaluate(latent_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Get predictions on the test set
y_pred_probs = classifier.predict(latent_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Find indices of correct and incorrect predictions
correct_indices = np.nonzero(y_pred_classes == y_test)[0]
incorrect_indices = np.nonzero(y_pred_classes != y_test)[0]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display correct predictions
plt.figure(figsize=(15, 3))
for i, idx in enumerate(correct_indices[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.suptitle("Examples of Correct Predictions")
plt.show()

# Display incorrect predictions
plt.figure(figsize=(15, 3))
for i, idx in enumerate(incorrect_indices[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.suptitle("Examples of Incorrect Predictions")
plt.show()
plt.tight_layout()

