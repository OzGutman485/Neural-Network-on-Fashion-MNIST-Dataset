# Neural Network on Fashion-MNIST Dataset

This project demonstrates the implementation of a neural network on the Fashion-MNIST dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OzGutman485/Neural-Network-on-Fashion-MNIST-Dataset/blob/main/Neural-Network-on-Fashion-MNIST-Dataset.ipynb)

## Project Description
Project completed as part of ML course at Bar-Ilan University.
This project implements neural networks for classifying handwritten digits (MNIST) and fashion items (Fashion-MNIST) using both pure NumPy and PyTorch approaches.

## Overview
The project consists of two main parts:

1. MNIST digit classification using a custom neural network built with NumPy
2. Fashion-MNIST classification using a CNN implemented in PyTorch

## Features
- Custom implementation of backpropagation and gradient descent
- Batch processing for efficient training
- Real-time visualization of training progress
- Model comparison tools (trained vs untrained)
- Interactive prediction visualization

## Model Architectures
### MLP (NumPy Implementation)
- Input Layer: 784 neurons (28x28 pixels)
- Hidden Layer: 128 neurons with sigmoid activation
- Output Layer: 10 neurons with softmax activation

### CNN (PyTorch Implementation)
- 3 Convolutional layers with ReLU activation
- Max pooling after each conv layer
- 2 Fully connected layers
- Dropout for regularization

## Results
- MLP on MNIST: ~94% accuracy
- CNN on Fashion-MNIST: ~89% accuracy

## Usage Instructions
1. Click the "Open In Colab" button above to open the project in Google Colab.
2. Run the cells in order to see the training process and results.

## Important Note
This project is designed to run in the Google Colab environment only.
