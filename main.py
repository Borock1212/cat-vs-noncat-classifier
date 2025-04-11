"""
Author: Oleksii Shevchenko
Project: Deep Neural Network from scratch in NumPy
Description: This project implements a two-layer and L-layer feedforward neural network 
             using NumPy only, with forward/backward propagation, cost computation, parameter updates,
             prediction and visualization.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import time
import random
from PIL import Image

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Activation functions and their derivatives

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# Load images from a given folder and assign a label

def load_images_by_filename(folder, max_images=None):
    images, labels = [], []
    count = 0

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder, filename)
            img = Image.open(path).resize((64, 64))
            img = np.array(img)

            if img.shape == (64, 64, 3):
                if "cat" in filename.lower():
                    label = 1
                elif "dog" in filename.lower():
                    label = 0
                else:
                    continue  # skip unknowns

                images.append(img)
                labels.append(label)
                count += 1
                if max_images and count >= max_images:
                    break
    return images, labels


# Initialize network parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# Linear forward step

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

# Linear activation forward

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

# L-layer forward propagation

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

# Compute cost function

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    return np.squeeze(cost)

# Linear backward step

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

# Linear activation backward

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

# L-layer backward propagation

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads

# Update network parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

# Predict outputs

def predict(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return predictions

# Train model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs

def show_misclassified(X, Y_true, Y_pred, num_images=5, image_size=(64, 64)):
    """
    Show a few misclassified images from the dataset.
    """
    misclassified_indices = np.where(Y_true != Y_pred)[1]
    print(f"Total misclassified: {len(misclassified_indices)}")

    if len(misclassified_indices) == 0:
        print("No misclassified images to show.")
        return

    random.shuffle(misclassified_indices)
    selected = misclassified_indices[:num_images]

    for i, idx in enumerate(selected):
        img_array = X[:, idx].reshape(*image_size, 3) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {int(Y_pred[0, idx])}\nTrue: {int(Y_true[0, idx])}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_path_cats = "datasets/train/cats"
    train_path_dogs = "datasets/train/dogs"
    test_path_cats = "datasets/test/cats"
    test_path_dogs = "datasets/test/dogs"

    # Load training images
    cat_train_images, cat_train_labels = load_images_by_filename(train_path_cats, max_images=500)
    dog_train_images, dog_train_labels = load_images_by_filename(train_path_dogs, max_images=500)

    # Load test images
    cat_test_images, cat_test_labels = load_images_by_filename(test_path_cats, max_images=100)
    dog_test_images, dog_test_labels = load_images_by_filename(test_path_dogs, max_images=100)

    # Combine training data
    X_train = np.array(cat_train_images + dog_train_images).astype(np.float32)
    Y_train = np.array(cat_train_labels + dog_train_labels).reshape(1, -1)

    # Combine test data
    X_test = np.array(cat_test_images + dog_test_images).astype(np.float32)
    Y_test = np.array(cat_test_labels + dog_test_labels).reshape(1, -1)

    # Shuffle training set
    perm_train = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm_train]
    Y_train = Y_train[:, perm_train]

    # Shuffle test set
    perm_test = np.random.permutation(X_test.shape[0])
    X_test = X_test[perm_test]
    Y_test = Y_test[:, perm_test]

    # Normalize and reshape
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.

    # Define network architecture
    layers_dims = [12288, 64, 32, 1]

    #start timer
    start_time = time.time()

    # Train the model
    parameters, costs = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.004, num_iterations=1500, print_cost=True)

    #end timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")

    # Make predictions
    predictions_train = predict(X_train, parameters)
    predictions_test = predict(X_test, parameters)

    # Accuracy
    print("\nTrain Accuracy: {:.2f}%".format(100 - np.mean(np.abs(predictions_train - Y_train)) * 100))
    print("Test Accuracy: {:.2f}%".format(100 - np.mean(np.abs(predictions_test - Y_test)) * 100))

    # Show misclassified test images
    show_misclassified(X_test, Y_test, predictions_test, image_size=(64, 64))

    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title('Learning rate = 0.0075')
    plt.show()