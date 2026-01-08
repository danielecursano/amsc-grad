# PINN with automatic differentiation

This repository contains a C++ implementation of a custom tensor class 
with **automatic differentiation up to second-order diagonal derivatives**,
along with core neural networks utilities and a **Physics-Informed
Neural Network (PINN)** applied to a **2D Laplace problem**.

## Index
- [Features](#features)
- [Example Problem: 2D Laplace Equation](#example-problem-2d-laplace-equation)
- [Usage](#usage)
- [Build Instructions](#build-instructions)
- [References](#references)



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

In `src/main.cpp`, the code solves the Laplace equation on the domain [-1, 1] × [-1, 1]:

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
\mathcal{L} =
\lambda_{\text{pde}} \ \mathcal{L}_{\text{pde}} +
\lambda_{\text{data}} \ \mathcal{L}_{\text{data}}
$$

where:
- $\mathcal{L}_{\text{pde}}$ enforces the Laplace equation in the interior
- $\mathcal{L}_{\text{data}}$ enforces the boundary conditions

### Results

After training, the model is evaluated on **N uniformly spaced grid points** over the domain.  
The resulting solution is visualized using the Python notebook  
`post_processing.ipynb`.


## Usage

To use the library, include the header

```cpp
#include "tensor.hpp"
```

## Build Instructions

The project is written in standard C++ and requires a compiler supporting **C++23** or newer.

### Build with OpenBLAS

From the root of the repository, compile with:

```bash
g++ -std=c++23 -I include/ -lopenblas -DUSE_BLAS src/main.cpp -o pinn
```

* `-DUSE_BLAS` enables the use of BLAS-backed `matmul`.

### Running the PINN Example

You can run the compiled program from the root of the repository:

```bash
./pinn [--epochs=<num_epochs>] [--lr=<learning_rate>] [--verbose]
```

Additional parameters can be set in the configuration file `pinn_config.dat`

### Optional Flags

* `-DDEBUG`
  Enables writing the backpropagation computation graph to `graph.dot`.
  You can then visualize it using the provided script:

  ```bash
  python render_graph.py graph.dot
  ```

## References

- Hubert Baty, A hands-on introduction to Physics-Informed Neural Networks for solving partial differential equations with benchmark tests taken from astrophysics and plasma physics. [arXiv:2403.00599v1](https://arxiv.org/html/2403.00599v1)  
- Karpathy, A. *micrograd*: Minimal Autograd Engine. [GitHub](https://github.com/karpathy/micrograd)  
- PINN material from the NAML course at Politecnico di Milano (Polimi)  
- Wikipedia: [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)  
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)  
- Autograd systems inspiration: [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/)
