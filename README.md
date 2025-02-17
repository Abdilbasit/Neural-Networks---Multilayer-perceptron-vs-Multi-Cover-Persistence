# Multiclass Perceptron & Multi-Layer Perceptron on Palmer Penguins

This project implements two custom classifiers—a Multiclass Perceptron (MCP) and a Multi-Layer Perceptron (MLP)—to classify species in the Palmer Penguins dataset. The project demonstrates various aspects of model training and evaluation including hyperparameter tuning, learning rate optimization, training sample size analysis, use of validation sets, cross-validation, and final performance assessment using confusion matrices and classification reports.

## Overview

The project is divided into two main parts:

- **MCP (Multiclass Perceptron):**  
  A simple neural network model using softmax activation for multi-class classification. The MCP is trained on standardized penguin features (bill length, bill depth, flipper length, and body mass) with experiments exploring:
  - Error rate evolution over epochs (Part A)
  - Optimal learning rate search (Part B)
  - Impact of varying training sample sizes (Part C)
  - Performance with a separate validation set (Part D)
  - K-Fold cross-validation analysis (Part E)
  - Detailed test set evaluation including confusion matrix and misclassified cases (Part F)

- **MLP (Multi-Layer Perceptron):**  
  A deeper network with a customizable hidden layer configuration using ReLU activation for hidden layers and softmax for the output. Similar experiments are performed:
  - Monitoring error rates during training (Part A)
  - Finding the optimal learning rate (Part B)
  - Studying the effect of training sample size (Part C)
  - Using a dedicated validation set (Part D)
  - Cross-validation (Part E)
  - In-depth performance analysis with confusion matrix and classification report (Part F)

## Features

- **Custom Implementation:**  
  Both the MCP and MLP are implemented from scratch using NumPy for matrix operations.
  
- **Comprehensive Experimentation:**  
  The project includes multiple experime
