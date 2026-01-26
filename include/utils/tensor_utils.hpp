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

    /**
     * Sets the seed for the random numbers generator.
     *
     * @param seed
     */
    inline void set_seed(uint32_t seed) {
        global_rng().seed(seed);
    }

    template<typename T, typename... Args>
    inline std::shared_ptr<Tensor<T>> make_tensor(Args &&... args) {
        return std::make_shared<Tensor<T>>(std::forward<Args>(args)...);
    }

    /**
     * @brief Creates a tensor with all elements initialized to zero.
     *
     * @tparam T
     * @param shape Shape of the tensor
     * @param requires_grad Whether the tensor should track gradients
     * @return Shared pointer to the new tensor
     */
    template<typename T>
    inline std::shared_ptr<Tensor<T>> zeros(const typename Tensor<T>::Shape &shape, bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), T(0));
        return make_tensor<T>(shape, data, requires_grad);
    }

    /**
     * @brief Creates a tensor with all elements initialized to one.
     *
     * @tparam T
     * @param shape Shape of the tensor
     * @param requires_grad Whether the tensor should track gradients
     * @return Shared pointer to the new tensor
     */
    template<typename T>
    inline std::shared_ptr<Tensor<T>> ones(const typename Tensor<T>::Shape &shape, bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), T(1));
        return make_tensor<T>(shape, data, requires_grad);
    }

    /**
     * @brief Creates a tensor with elements initialized from a uniform distribution.
     *
     * Each element is independently sampled from a uniform distribution
     * in the range [\p low, \p high).
     *
     * @tparam T Numeric type
     * @param shape Shape of the tensor (vector of dimension sizes)
     * @param low Lower bound of the uniform distribution (inclusive, default 0)
     * @param high Upper bound of the uniform distribution (exclusive, default 1)
     * @param requires_grad Whether the tensor should track gradients (default: false)
     * @return Shared pointer to a tensor of the specified shape with uniform data
     */
    template<typename T>
    inline std::shared_ptr<Tensor<T>>
    uniform(const typename Tensor<T>::Shape &shape, T low = T(0), T high = T(1), bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
        std::uniform_real_distribution<T> dist(low, high);
        auto &gen = global_rng();
        for (auto &x: data) x = dist(gen);
        return make_tensor<T>(shape, data, requires_grad);
    }

    /**
     * @brief Creates a tensor with elements initialized from a uniform distribution.
     *
     * Each element is independently sampled from a uniform distribution
     * in the range [\p low, \p high).
     *
     * @tparam T Numeric type
     * @param shape Shape of the tensor (vector of dimension sizes)
     * @param mean Mean of the distribution
     * @param stddev Standard deviation of the distribution
     * @param requires_grad Whether the tensor should track gradients (default: false)
     * @return Shared pointer to a tensor of the specified shape with uniform data
     */
    template<typename T>
    inline std::shared_ptr<Tensor<T>>
    normal(const typename Tensor<T>::Shape &shape, T mean = T(0), T stddev = T(1), bool requires_grad = false) {
        std::vector<T> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
        std::normal_distribution<T> dist(mean, stddev);
        auto &gen = global_rng();
        for (auto &x: data) x = dist(gen);
        return make_tensor<T>(shape, data, requires_grad);
    }

    /**
     * @brief Creates a identity square matrix.
     */
    template <typename T>
    inline std::shared_ptr<Tensor<T>> 
    eye(const typename Tensor<T>::Shape& shape, bool requires_grad = false) {
        if (shape.size() != 2) throw std::runtime_error("eye() requires a 2D tensor shape");
        if (shape[0] != shape[1]) throw std::runtime_error("Tensor must be a square matrix");
        const auto n = shape[0];
        std::vector<T> data(n*n, T(0));
        for (auto i = 0; i < n; ++i) {
            data[i*n + i] = T{1};
        }
        return make_tensor<T>(shape, data, requires_grad);
    }

    /**
     * @brief Generate a random permutation of indices [0, n).
     *
     * Returns a vector containing a uniformly random permutation of the integers
     * from 0 to n-1. The permutation is generated using the Fisher–Yates shuffle
     * and the global random number generator.
     *
     * @param n Number of elements in the permutation.
     * @return A vector of size n containing a random permutation of [0, n).
     */
    inline std::vector<size_t> random_perm(size_t n) {
        std::vector<size_t> perm(n);
        for (size_t i = 0; i < n; ++i) perm[i] = i;

        auto& gen = global_rng();
        for (size_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            std::swap(perm[i], perm[dist(gen)]);
        }
        return perm;
    }

}

#endif
