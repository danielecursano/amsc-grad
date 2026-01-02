#ifndef ADAM_HPP
#define ADAM_HPP

#include "optim/optim.hpp"
#include <cmath>

namespace tensor::optim {

    /**
     * @brief Data structure to store a tensor and its optimizer state for Adam.
     *
     * Stores the first moment (m), second moment (v), and whether weight decay
     * should be applied for a given parameter tensor.
     *
     * @tparam T Numeric type
     */
    template<Numeric T>
    struct AdamVariable {

        /**
         * @return the number of elements in the parameter tensor.
         */
        size_t size() {
            return m.size();
        }

        AdamVariable(TensorS<T> tensor, bool decay) :
                tensor(tensor),
                decay(decay),
                m(tensor->data.size(), T(0)),
                v(tensor->data.size(), T(0)) {}

        /// Tensor being optimized
        TensorS<T> tensor;

        /// First moment vector
        std::vector<T> m;

        ///Second moment vector
        std::vector<T> v;

        /// Wheter to apply weight decay
        bool decay;
    };

    /**
     * @brief Adam optimizer.
     *
     * Implements the Adam optimization algorithm.
     *
     * References:
     * \link https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
     * \link https://arxiv.org/abs/1412.6980
     *
     * @tparam T
     */
    template<Numeric T>
    class Adam : public Optimizer<T> {
    public:
        Adam(const std::vector<AdamVariable<T>> &params, T learning_rate, T beta1, T beta2, T eps, T weight_decay)
                : params(params), lr(learning_rate), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay),
                  step_count(0) {}

        void step() override {
            step_count++;
            T step_size = this->lr * std::sqrt((1 - std::pow(beta2, step_count))) / (1 - std::pow(beta1, step_count));
            for (auto &p: this->params) {
                for (size_t i = 0; i < p.size(); ++i) {
                    T grad = (p.tensor->grad)[i];
                    if (p.decay) grad += weight_decay * (p.tensor->data)[i];
                    p.m[i] = beta1 * p.m[i] + (1.0 - beta1) * grad;
                    p.v[i] = beta2 * p.v[i] + (1.0 - beta2) * grad * grad;
                    (p.tensor->data)[i] -= step_size * p.m[i] / (std::sqrt(p.v[i]) + eps);
                }
            }
        }

        void zero_grad() override {
            for (auto &p: this->params) {
                p.tensor->zero_grad();
            }
        }

    private:
        /// Parameters being optimized
        std::vector<AdamVariable<T>> params;

        /// Learning rate
        T lr;

        /// First moment decay
        T beta1;

        /// Second moment decay
        T beta2;

        /// Small epsilon to prevent division by zero
        T eps;

        /// Weight decay
        T weight_decay;

        /// Step counter
        int step_count = 0;
    };

}
#endif