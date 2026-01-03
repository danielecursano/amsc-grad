# PINN with automatic differentiation

This repository contains a C++ implementation of a custom tensor class 
with **automatic differentiation up to second-order diagonal derivatives**,
along with core neural networks utilities and a **Physics-Informed
Neural Network (PINN)** applied to a **2D Laplace problem**.



## Features

### Tensor Class
- Dynamic tensor structure with basic linear algebra support
- Automatic differentiation (reverse-mode)
- Support for first-order derivatives and second-order diagonal derivatives

### Operations
- Element-wise operations
- Matrix multiplication
- Activation functions
- Gradient accumulation and backpropagation

### Optimizer
- **Adam optimizer** 
- Stochastic gradient descent

## Example Problem: 2D Laplace Equation

In `src/main.cpp`, the code solves the Laplace equation on the domain  
[-1, 1] × [-1, 1]:

$$
\nabla^2 u(x, y) = 0
$$

with **Dirichlet boundary conditions** given by:

$$
u(x, y) = x^2 - y^2 \quad \text{on } \partial \Omega
$$

### Problem Setup

- A random dataset is generated consisting of:
  - **400 interior points**
  - **120 boundary points**
- A fully connected neural network is constructed with:
  - **5 hidden dense layers**
  - **20 neurons per layer**
  - **tanh** activation function
- The network is trained to minimize the combined loss:

$$
\mathcal{L}
=
\lambda_{\text{pde}} \, \mathcal{L}_{\text{pde}}
+
\lambda_{\text{data}} \, \mathcal{L}_{\text{data}}
$$

where:
- $\mathcal{L}_{\text{pde}}$ enforces the Laplace equation in the interior
- $\mathcal{L}_{\text{data}}$ enforces the boundary conditions

### Results

After training, the model is evaluated on **N uniformly spaced grid points** over the domain.  
The resulting solution is visualized using the Python notebook  
`post_processing.ipynb`.
