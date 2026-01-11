#ifndef TENSOR_CORE_HPP
#define TENSOR_CORE_HPP

#ifdef DEBUG
    #define TENSOR_DEBUG 1
#else
    #define TENSOR_DEBUG 0
#endif

#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <cmath>
#include <string>

#include "defines.hpp"

template<Numeric T> struct Tensor;
template<Numeric T> using TensorS = std::shared_ptr<Tensor<T>>;

/**
 * @brief Multi dimensional container for data and automatic differentiation.
 *
 * This class represents a node in the computational graph. It stores
 * the forward-pass data, the accumulated gradient and Hessian, a function pointer to the backward
 * function for gradient propagation
 *
 * @tparam T the numeric type (e.g., float, double)
 */
template<Numeric T>
struct Tensor : public std::enable_shared_from_this<Tensor<T>> {

    using Shape = std::vector<size_t>;

    /// Data values
    std::vector<T> data;

    /// First-order gradients
    std::vector<T> grad;

    /// Second-order derivatives
    std::vector<T> hess;

    /// Shape of the tensor
    Shape shape;

    /// true if the tensor requires the gradient computation
    bool requires_grad;

    /// Parent tensors in the computational graph
    std::vector<TensorS<T>> prev;

    /// Backward function for gradient propagation
    std::function<void()> grad_fn = []() {};

    /// Optional metadata (e.g. operation name)
    std::string metadata = "";

    /**
     * @brief Constructs a tensor.
     *
     * @param shape Shape of the tensor
     * @param data Initial tensor data
     * @param requires_grad Whether gradients should be tracked
     * @param parents Parent tensors in the computational graph
     * @param metadata Optional metadata string
     */
    Tensor(
            Shape shape,
            std::vector<T> data,
            bool requires_grad = false,
            std::vector<std::shared_ptr<Tensor<T>>> parents = {},
            std::string metadata = ""
        ) : shape(std::move(shape)),
            requires_grad(requires_grad),
            data(std::move(data)),
            prev(std::move(parents)),
            metadata(metadata),
            grad(requires_grad ? this->data.size() : 0),
            hess(requires_grad ? this->data.size() : 0)
         {
            size_t total_size = 1;
            for (auto dim: shape) total_size *= (dim > 0 ? dim : 1);
            if (this->data.empty()) this->data.assign(total_size, T(0));
            // NOTE: If data starts empty, grad/hess were sized before auto-filling data.
            // Fix: resize grad/hess after data is assigned (or size by total_size).
         }


    /**
     * @brief Performs backpropagation starting from this tensor.
     *
     * Builds a topological ordering of the computation graph
     * and executes each node's gradient function in reverse order.
     */
    void backward()
    {
        std::vector<TensorS<T>> graph;
        std::unordered_set<Tensor<T>*> visited;

        std::function<void(TensorS<T>)> build_graph = [&](TensorS<T> v) {
            if (visited.find(v.get()) == visited.end()) {
                visited.insert(v.get());
                for (auto &p: v->prev) build_graph(p);
                graph.push_back(v);
            }
        };

        build_graph(this->shared_from_this());

        if (this->requires_grad) {
            // NOTE: This seeds every output element with grad=1, i.e., gradients of sum(outputs).
            // Fix: accept an upstream gradient or require scalar outputs to avoid surprises.
            std::fill(this->grad.begin(), this->grad.end(), T(1));
            std::fill(this->hess.begin(), this->hess.end(), T(0));
        }

        #if TENSOR_DEBUG
        print_graph(graph);
        #endif

        for (auto it = graph.rbegin(); it != graph.rend(); ++it) {
            (*it)->grad_fn();
        }

        // Breaks links to parent nodes and clears grad_fn, freeing temporary nodes after backward.
        for (auto &node: graph) {
            node->prev.clear();
            node->grad_fn = []() {};
        }
        // NOTE: Clearing the graph prevents repeated backward() on the same forward pass.
        // Fix: make graph cleanup optional if multiple backward passes are needed.

    }

    /**
     * @brief Resets gradients and Hessians to zero.
     */
    void zero_grad()
    {
        std::fill(this->grad.begin(), this->grad.end(), T(0));
        std::fill(this->hess.begin(), this->hess.end(), T(0));
    }

};

#endif
