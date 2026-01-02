#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

#include <memory>
#include <vector>
#include <random>
#include "core/tensor_core.hpp"

namespace tensor {

    inline std::mt19937 &global_rng() {
        static std::mt19937 gen{std::random_device{}()};
        return gen;
    }

    inline void set_seed(uint32_t seed) {
        global_rng().seed(seed);
    }

    template<typename T, typename... Args>
    inline std::shared_ptr<Tensor<T>> make_tensor(Args &&... args) {
        return std::make_shared<Tensor<T>>(std::forward<Args>(args)...);
    }

    template<typename T>
    inline std::shared_ptr<Tensor<T>> zeros(const typename Tensor<T>::Shape &shape, bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), T(0));
        return make_tensor<T>(shape, data, requires_grad);
    }

    template<typename T>
    inline std::shared_ptr<Tensor<T>> ones(const typename Tensor<T>::Shape &shape, bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), T(1));
        return make_tensor<T>(shape, data, requires_grad);
    }

    template<typename T>
    inline std::shared_ptr<Tensor<T>>
    uniform(const typename Tensor<T>::Shape &shape, T low = T(0), T high = T(1), bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
        std::uniform_real_distribution<T> dist(low, high);
        auto &gen = global_rng();
        for (auto &x: data) x = dist(gen);
        return make_tensor<T>(shape, data, requires_grad);
    }

    template<typename T>
    inline std::shared_ptr<Tensor<T>>
    normal(const typename Tensor<T>::Shape &shape, T mean = T(0), T stddev = T(1), bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
        std::normal_distribution<T> dist(mean, stddev);
        auto &gen = global_rng();
        for (auto &x: data) x = dist(gen);
        return make_tensor<T>(shape, data, requires_grad);
    }
}

#endif
