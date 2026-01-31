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

struct TensorMetadata {
    std::string name;
    std::string grad_function_name;

    TensorMetadata(std::string name, std::string grad_function_name) :
    name(name),
    grad_function_name(grad_function_name) {}

};

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
    using shape_type = size_t;
    using Shape = std::vector<shape_type>;

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
    TensorMetadata metadata;

    /**
     * @brief Constructs a tensor.
     *
     * @param shape Shape of the tensor
     * @param data Initial tensor data
     * @param requires_grad Whether gradients should be tracked
     * @param parents Parent tensors in the computational graph
     * @param metadata Optional metadata strings
     */
    Tensor(
            Shape shape,
            std::vector<T> data,
            bool requires_grad = false,
            std::vector<std::shared_ptr<Tensor<T>>> parents = {},
            std::string grad_function_name = "",
            std::string name = ""
        ) : shape(std::move(shape)),
            requires_grad(requires_grad),
            data(std::move(data)),
            prev(std::move(parents)),
            grad(requires_grad ? this->data.size() : 0),
            hess(requires_grad ? this->data.size() : 0),
            metadata(name, grad_function_name)
         {
            size_t total_size = 1;
            for (auto dim: shape) total_size *= (dim > 0 ? dim : 1);
            if (this->data.empty()) this->data.assign(total_size, T(0));
         }


    /**
     * @brief Performs backpropagation starting from this tensor.
     *
     * Builds a topological ordering of the computation graph
     * and executes each node's gradient function in reverse order.
     * 
     * @param clean_graph if true it cleans the tensor's parents vector and 
     *                    gradient function to free memory
     */
    void backward(bool clean_graph = true)
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
        if (clean_graph) {
            for (auto &node: graph) {
                node->prev.clear();
                node->grad_fn = []() {};
            }
        }

    }

    /**
     * @brief Resets gradients and Hessians to zero.
     */
    void zero_grad()
    {
        std::fill(this->grad.begin(), this->grad.end(), T(0));
        std::fill(this->hess.begin(), this->hess.end(), T(0));
    }

    /**
     * @brief Permutes the rows of a 2D tensor.
     * 
     * Reorderd the rows of the tensor inplace according to the 
     * given permutation vector.
     * 
     * @param perm A permutation vector of size equal to the number of rows.
     *             Each value must be a valid row index.
     * 
     */
    void permute_rows(const std::vector<size_t>& perm)
    {
        if (shape.size() != 2)
            throw std::runtime_error("permute_rows_ requires a 2D tensor");

        const shape_type N = shape[0];
        const shape_type M = shape[1];

        std::vector<bool> visited(N, false);
        std::vector<T> temp_row(M);

        for (size_t i = 0; i < N; ++i) {
            if (visited[i]) continue;

            size_t current = i;
            std::copy(
                data.begin() + i * M,
                data.begin() + (i + 1) * M,
                temp_row.begin()
            );

            while (!visited[current]) {
                visited[current] = true;
                size_t next = perm[current];

                if (next == i) {
                    std::copy(
                        temp_row.begin(),
                        temp_row.end(),
                        data.begin() + current * M
                    );
                } else {
                    std::copy(
                        data.begin() + next * M,
                        data.begin() + (next + 1) * M,
                        data.begin() + current * M
                    );
                }

                current = next;
            }
        }
    }

};

#endif