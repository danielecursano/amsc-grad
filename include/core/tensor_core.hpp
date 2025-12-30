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

/*
 * Multi dimensional container for data and its gradient.
 * This class represents a node in the computational graph. It stores
 * the forward-pass data, the accumulated gradient, a function pointer to the backward
 * function to propagate gradients from the loss to the input layers during backpropagation.
 * 
 * @tparam T the numeric type (e.g., float, double)
 */
template<Numeric T>
struct Tensor : public std::enable_shared_from_this<Tensor<T>> {
    using Shape = std::vector<int>; // using integers to then implement -1 as a placeholder ?

    std::vector<T> data, grad, hess;
    Shape shape;

    bool requires_grad;

    std::vector<TensorS<T>> prev;
    std::function<void()> grad_fn = []() {};

    std::string metadata = "";

    Tensor(
            Shape shape, 
            std::vector<T> data, 
            bool requires_grad = true, 
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
         }
        
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
            std::fill(this->grad.begin(), this->grad.end(), T(1));
            std::fill(this->hess.begin(), this->hess.end(), T(0));
        }

        #if TENSOR_DEBUG
        print_graph(graph);
        #endif 

        for (auto it = graph.rbegin(); it != graph.rend(); ++it) {
            (*it)->grad_fn();
        }
    }

    void zero_grad() 
    {
        std::fill(this->grad.begin(), this->grad.end(), T(0));
        std::fill(this->hess.begin(), this->hess.end(), T(0));
    }

};

#endif