#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "core/tensor_core.hpp"
#include "defines.hpp"
#include <vector>
#include <memory>

namespace tensor::optim {

    /**
     * @brief Abstract base class for optimizers.
     *
     * This class defines the interface for all optimization algorithms.
     * Optimizers are used for updating parameters of tensors based on
     * their gradients during training.
     *
     * @tparam T Numeric type of the paramteres
     */
    template<Numeric T>
    class Optimizer {
    public:

        virtual ~Optimizer() = default;

        /**
         * @brief Performs a single optimization step.
         *
         * Derived class implement this function to update parameters
         * according to their specific optimization algorithms.
         */
        virtual void step() = 0;

        /**
         * Resets gradients of all parameters to zero.
         */
        virtual void zero_grad() = 0;

    };

    /**
     * @brief Stochastic Gradient Descent (SGD) optimizer.
     *
     * Updates parameters using the standard gradient descent rule.
     *
     * Reference:
     * \link https://en.wikipedia.org/wiki/Stochastic_gradient_descent
     *
     */
    template<Numeric T>
    class SGD : public Optimizer<T> {
    public:
        SGD(const std::vector<TensorS<T>> &params, T learning_rate)
                : params(params), lr(learning_rate) {}

        void step() override {
            for (auto &p: this->params) {
                for (size_t i = 0; i < p->data.size(); ++i)
                    p->data[i] -= this->lr * p->grad[i];
            }
        }

        void zero_grad() override {
            for (auto &p: this->params) {
                p->zero_grad();
            }
        }

    private:
        /// Vector of parameters to optimize
        std::vector<TensorS<T>> params;

        /// Learning rate
        T lr;
    };

}

#endif