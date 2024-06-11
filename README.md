# Scalargrad

Scalargrad is a compact and efficient implementation of an Autograd engine and a miniature neural network library, inspired by Karpathy's micrograd. This project demonstrates the core principles of automatic differentiation and backpropagation through a novel approach of operating on scalar values.

## Overview

Unlike traditional frameworks that handle tensors or matrices, Scalargrad breaks down each neuron into its individual scalar operations, such as additions and multiplications. By dynamically constructing a computational graph (DAG) of these scalar operations, Scalargrad enables efficient reverse-mode automatic differentiation, allowing for precise calculation of gradients.

Built on top of the Autograd engine, Scalargrad provides a PyTorch-like API for defining and training small neural networks for binary classification tasks. Despite its compact codebase of around 100 lines for the Autograd engine and 50 lines for the neural network library, Scalargrad showcases the power and flexibility of automatic differentiation.

## Features

- **Scalar-based Computations**: Operates on scalar values, breaking down neurons into individual operations like additions and multiplications.
- **Autograd Engine**: Implements efficient reverse-mode automatic differentiation through a dynamically constructed computational graph (DAG).
- **Neural Network Library**: Provides a PyTorch-like API for defining and training small neural networks for binary classification tasks.
- **Compact Codebase**: Implemented in around 150 lines of code (100 for Autograd, 50 for the neural network library).
- **Educational Resource**: Offers an insightful and accessible exploration of automatic differentiation and neural network concepts.

