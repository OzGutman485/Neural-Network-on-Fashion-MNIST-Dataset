# Neural Network on Fashion-MNIST Dataset

This project demonstrates the implementation of a neural network on the Fashion-MNIST dataset.

## Project Description

Project completed as part of the ML course at Bar-Ilan University. This project implements neural networks for classifying handwritten digits (MNIST) and fashion items (Fashion-MNIST) using both pure NumPy and PyTorch approaches.

## Overview

The project consists of three main parts:

1. **MLP (NumPy Implementation)**: Handwritten digit classification (MNIST) using a custom neural network.
2. **Neural Network (PyTorch Implementation)**: Fashion-MNIST classification with a simple neural network.
3. **CNN (PyTorch Implementation)**: Convolutional neural network applied to the Fashion-MNIST dataset.

## Features

* Custom implementation of backpropagation and gradient descent.
* Batch processing for efficient training.
* Real-time visualization of training progress.
* Model comparison tools (trained vs untrained).
* Interactive prediction visualization.

## Model Architectures

### Part 1: MLP (NumPy Implementation)
* Input Layer: 784 neurons (28x28 pixels)
* Hidden Layer: 128 neurons with sigmoid activation
* Output Layer: 10 neurons with softmax activation

### Part 2: Neural Network (PyTorch Implementation)
* Input Layer: 784 neurons (28x28 pixels)
* Hidden Layer: 128 neurons with ReLU activation
* Output Layer: 10 neurons with softmax activation

### Part 3: CNN (PyTorch Implementation)
* 3 Convolutional layers with ReLU activation
* Max pooling after each conv layer
* 2 Fully connected layers
* Dropout for regularization

## Results

* **MLP on MNIST**: ~96% accuracy
* **Neural Network on Fashion-MNIST**: ~80% accuracy
* **CNN on Fashion-MNIST**: ~83% accuracy

## Usage Instructions

1. Click the "Open In Colab" button above to open the project in Google Colab.
2. Run the cells in order to see the training process and results.

## Important Note

This project is designed to run in the Google Colab environment only.
