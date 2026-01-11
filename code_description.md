# Code Overview and Potential Issues

This report describes the code in this project and highlights possible errors or drawbacks.

## Repository structure

- `include/`: Header-only tensor library with automatic differentiation, ops, optimizers, and utilities.
- `src/main.cpp`: PINN demo solving 2D Laplace with the tensor library.
- `examples/`: Small training example using a linear model and Adam.
- `tests/`: Minimal tests for activations, matmul, and gradients.
- `render_graph.py` and `post_processing.ipynb`: Visualization helpers for debugging and output plots.

## Core library details

### Tensor core (`include/core/tensor_core.hpp`)

- `Tensor<T>` stores `data`, `grad`, `hess`, `shape`, and the computation graph (`prev`, `grad_fn`).
- `backward()` builds a topological order of the graph, sets the output gradient and Hessian, and runs each node's `grad_fn` in reverse.
- `zero_grad()` clears `grad` and `hess` buffers.

### Operations

- `include/ops/arithmetic.hpp`:
  - Overloaded `operator+`, `operator*` for elementwise and scalar multiplications.
  - `pow`, `sum`, `mean`, and `broadcast_add`.
  - Backward rules propagate gradients and diagonal Hessians.
- `include/ops/activations.hpp`:
  - `relu` and `tanh`, with gradient and Hessian propagation.
- `include/ops/matmul.hpp`:
  - `raw_matmul` with BLAS backend (if enabled) or a fallback O(mnp) implementation.
  - `matmul` for 2D tensors and backward rules for `grad` and `hess`.

### Optimizers (`include/optim/*.hpp`)

- `Optimizer` base class defines `step()` and `zero_grad()`.
- `SGD` applies simple gradient descent.
- `Adam` tracks per-parameter first/second moment estimates and supports weight decay.

### Utilities

- `include/utils/tensor_utils.hpp`: RNG, seed control, and helper tensor constructors (`zeros`, `ones`, `uniform`, `normal`).
- `include/utils/debug.hpp`: `operator<<` for tensor metadata and `print_graph` to dump a Graphviz `.dot` file.
- `include/defines.hpp`: `Numeric` concept (floating-point only).

## Applications

### PINN example (`src/main.cpp`)

- Builds a 5-layer fully connected network with `tanh` activations.
- Uses `x->hess` to build a Laplacian term for PDE loss.
- Combines PDE loss and boundary loss, then trains with Adam.
- Writes `history.csv` and `output.csv` for analysis.

### Small NN example (`examples/small_nn.cpp`)

- Linear model trained with Adam on a small synthetic dataset.

### Tests (`tests/*.cpp`)

- `test_grad.cpp` checks arithmetic, mean, and tanh example.
- `test_matmul.cpp` checks forward and backward gradients of matmul.
- `test_activations.cpp` checks ReLU (tanh test is incomplete).

## Possible errors or drawbacks

1. **Tensor gradient/Hessian size bug for empty `data`**
   - In `Tensor` constructor, `grad` and `hess` are sized using `data.size()` before `data` is potentially auto-filled when empty.
   - If a tensor is constructed with empty `data` and `requires_grad=true`, its `grad`/`hess` vectors remain size 0 even after `data` is resized.
   - File: `include/core/tensor_core.hpp`.

2. **`backward()` assumes a unit gradient for every output element**
   - `Tensor::backward()` sets all output gradients to 1 (not just a scalar loss).
   - For non-scalar outputs, this computes gradients for the sum of outputs, which might not match user intent.
   - File: `include/core/tensor_core.hpp`.

3. **Computation graph is cleared after each backward pass**
   - After `backward()`, `prev` and `grad_fn` are cleared for all nodes.
   - This makes repeated backward passes on the same graph impossible without recomputing the forward pass.
   - File: `include/core/tensor_core.hpp`.

4. **`broadcast_add` assumes 2D tensors but does not enforce `a` shape length**
   - The function checks `b` but assumes `a->shape[0]` and `a->shape[1]` exist.
   - If `a` is not 2D, this will read out of bounds.
   - File: `include/ops/arithmetic.hpp`.

5. **Activation tests are incomplete**
   - `tests/test_activations.cpp` computes `tanh` forward/backward but does not assert correctness.
   - This reduces confidence in the activation gradients/Hessians.
   - File: `tests/test_activations.cpp`.

6. **Potential numeric stability issues for second derivatives**
   - The PINN example uses `float` and computes Hessians, which can be noisy or unstable for higher-order derivatives.
   - Using `double` for second derivatives may improve stability.
   - File: `src/main.cpp`.

7. **`matmul` backward relies on external gradient zeroing**
   - Backward routines in `matmul` and arithmetic ops accumulate into existing `grad`/`hess` buffers.
   - If callers forget to `zero_grad()` between steps, gradients will silently accumulate.
   - Files: `include/ops/matmul.hpp`, `include/ops/arithmetic.hpp`.

## Notes on limitations

- Only diagonal second derivatives are tracked; cross-partials are not supported.
- Only 2D matrices are supported for `matmul`.
- No broadcasting beyond `broadcast_add(?, (1,K))`.
- No general tensor slicing, reshaping, or reduction ops beyond `sum` and `mean`.

