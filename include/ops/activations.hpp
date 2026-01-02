#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "core/tensor_core.hpp"
#include <memory>

namespace tensor::ops {

    template<Numeric T>
    TensorS<T> relu(TensorS<T> a) {
        std::vector<T> out_data(a->data.size());
        std::transform(a->data.begin(), a->data.end(), out_data.begin(), [](T x) { return x > 0 ? x : 0; });

        auto out = std::make_shared<Tensor<T>>(
                a->shape,
                out_data,
                a->requires_grad,
                std::vector<TensorS<T>>{a},
                "ReLuBackward"
        );

        out->grad_fn = [a, out]() {
            if (a->requires_grad) {
                for (size_t i = 0; i < a->data.size(); ++i) {
                    T mask = (out->data[i] > 0 ? 1 : 0);
                    a->grad[i] += mask * out->grad[i];
                    a->hess[i] += mask * out->hess[i];
                }
            }
        };

        return out;
    }

    template<Numeric T>
    TensorS<T> tanh(TensorS<T> a) {
        std::vector<T> out_data(a->data.size());
        std::transform(a->data.begin(), a->data.end(), out_data.begin(), [](T x) { return std::tanh(x); });

        auto out = std::make_shared<Tensor<T>>(
                a->shape,
                out_data,
                a->requires_grad,
                std::vector<TensorS<T>>{a},
                "TanhBackward"
        );

        out->grad_fn = [a, out]() {
            if (a->requires_grad) {
                for (size_t i = 0; i < a->data.size(); ++i) {
                    T dy = 1 - out->data[i] * out->data[i];
                    T d2y = -2 * out->data[i] * dy;

                    a->grad[i] += out->grad[i] * dy;
                    a->hess[i] += out->hess[i] * dy * dy + out->grad[i] * d2y;
                }
            }
        };

        return out;
    }

}

#endif