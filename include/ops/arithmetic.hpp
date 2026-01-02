#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include "core/tensor_core.hpp"
#include <cmath>
#include <numeric>

namespace tensor::ops {

        template <Numeric T>
        TensorS<T> operator+(TensorS<T> a, TensorS<T> b)
        {
            if (a->shape != b->shape) throw std::runtime_error("Tensors shapes do not match");
            std::vector<T> out_data(a->data.size());

            for (size_t i = 0; i < a->data.size(); ++i) out_data[i] = a->data[i] + b->data[i];

            auto out = std::make_shared<Tensor<T>>(
                    a->shape,
                    out_data,
                    a->requires_grad || b->requires_grad,
                    std::vector<TensorS<T>>{a, b},
                    "AddBackward"
            );

            out->grad_fn = [a, b, out]() {
                if (a->requires_grad) {
                    std::transform(a->grad.begin(), a->grad.end(), out->grad.begin(), a->grad.begin(),
                                   [](T x, T y) { return x + y; });
                    std::transform(a->hess.begin(), a->hess.end(), out->hess.begin(), a->hess.begin(),
                                   [](T x, T y) { return x + y; });
                }
                if (b->requires_grad) {
                    std::transform(b->grad.begin(), b->grad.end(), out->grad.begin(), b->grad.begin(),
                                   [](T x, T y) { return x + y; });
                    std::transform(b->hess.begin(), b->hess.end(), out->hess.begin(), b->hess.begin(),
                                   [](T x, T y) { return x + y; });
                }
            };

            return out;
        }

        template <Numeric T>
        TensorS<T> operator*(TensorS<T> a, T scalar)
        {
            std::vector<T> out_data(a->data.size());
            std::transform(a->data.begin(), a->data.end(), out_data.begin(), [scalar](T x) { return x * scalar; });

            auto out = std::make_shared<Tensor<T>>(
                    a->shape,
                    out_data,
                    a->requires_grad,
                    std::vector<TensorS<T>>{a},
                    "MulScalarBackward"
            );

            out->grad_fn = [a, scalar, out]() {
                if (a->requires_grad) {
                    std::transform(a->grad.begin(), a->grad.end(), out->grad.begin(), a->grad.begin(),
                                   [scalar](T x, T y) { return x + y * scalar; });
                    std::transform(a->hess.begin(), a->hess.end(), out->hess.begin(), a->hess.begin(),
                                   [scalar](T x, T y) { return x + y * scalar * scalar; });
                }
            };

            return out;
        }

        template <Numeric T>
        TensorS<T> operator*(T scalar, TensorS<T> a)
        {
            return a * scalar;
        }

        template<Numeric T>
        TensorS<T> operator*(TensorS<T> a, TensorS<T> b)
        {
            if (a->shape != b->shape) throw std::runtime_error("Tensors shapes do not match");
            std::vector<T> out_data(a->data.size());
            std::transform(a->data.begin(), a->data.end(), b->data.begin(), out_data.begin(),
                           [](T x, T y) { return x * y; });

            auto out = std::make_shared<Tensor<T>>(
                    a->shape,
                    out_data,
                    a->requires_grad || b->requires_grad,
                    std::vector<TensorS<T>>{a, b},
                    "MulBackward"
            );

            out->grad_fn = [a, b, out]() {
                if (a->requires_grad) {
                    for (size_t i = 0; i < a->grad.size(); ++i) {
                        a->grad[i] += out->grad[i] * b->data[i];
                        a->hess[i] += out->hess[i] * b->data[i] * b->data[i];
                    }
                }
                if (b->requires_grad) {
                    for (size_t i = 0; i < a->grad.size(); ++i) {
                        b->grad[i] += out->grad[i] * a->data[i];
                        b->hess[i] += out->hess[i] * a->data[i] * a->data[i];
                    }
                }
            };

            return out;
        }

        template <Numeric T>
        TensorS<T> pow(TensorS<T> a, int exp)
        {
            std::vector<T> out_data(a->data.size());
            std::transform(a->data.begin(), a->data.end(), out_data.begin(), [exp](T x) { return std::pow(x, exp); });
            auto out = std::make_shared<Tensor<T>>(
                    a->shape,
                    out_data,
                    a->requires_grad,
                    std::vector<TensorS<T>>{a},
                    "PowBackward"
            );

            out->grad_fn = [a, out, exp]() {
                if (a->requires_grad) {
                    for (size_t i = 0; i < a->grad.size(); ++i) {
                        T fp = exp * std::pow(a->data[i], exp - 1);
                        T fpp = exp * (exp - 1) * std::pow(a->data[i], exp - 2);
                        a->grad[i] += out->grad[i] * fp;
                        a->hess[i] += out->hess[i] * fp * fp + out->grad[i] * fpp;
                    }
                }
            };

            return out;
        }

        template <Numeric T>
        TensorS<T> sum(TensorS<T> a) {
            std::vector<T> out_data(1);
            for (auto &val: a->data) out_data[0] += val;

            auto out = std::make_shared<Tensor<T>>(
                    typename Tensor<T>::Shape{1},
                    out_data,
                    a->requires_grad,
                    std::vector<TensorS<T>>{a},
                    "SumBackward"
            );

            out->grad_fn = [a, out]() {
                if (!a->requires_grad) return;
                for (size_t i = 0; i < a->data.size(); ++i) {
                    a->grad[i] += out->grad[0];
                    a->hess[i] += out->hess[0];
                }
            };

            return out;
        }

        template <Numeric T>
        TensorS<T> mean(TensorS<T> a) {
            return sum(a) * static_cast<T>(1. / static_cast<T>(a->data.size()));
        }

        template <Numeric T>
        TensorS<T> broadcast_add(TensorS<T> a, TensorS<T> b)
        {
            if (b->shape[0] != 1 || b->shape[1] != a->shape[1]) {
                throw std::runtime_error("broadcast_add expects b to have shape (1, K)");
            }

            size_t N = a->shape[0];
            size_t K = a->shape[1];

            std::vector<T> out_data(N * K);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    out_data[i * K + j] = a->data[i * K + j] + b->data[j];
                }
            }

            auto out = std::make_shared<Tensor<T>>(
                    typename Tensor<T>::Shape{N, K},
                    out_data,
                    a->requires_grad || b->requires_grad,
                    std::vector<TensorS<T>>{a, b},
                    "BroadcastAddBackward"
            );

            out->grad_fn = [a, b, out, N, K]() {
                if (a->requires_grad) {
                    for (size_t i = 0; i < N * K; ++i) {
                        a->grad[i] += out->grad[i];
                        a->hess[i] += out->hess[i];
                    }
                }
                if (b->requires_grad) {
                    for (size_t i = 0; i < N; ++i) {
                        for (size_t j = 0; j < K; ++j) {
                            b->grad[j] += out->grad[i * K + j];
                            b->hess[j] += out->hess[i * K + j];
                        }
                    }
                }
            };

            return out;
        }

}

#endif