#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "core/tensor_core.hpp"
#include "defines.hpp"
#include <vector>
#include <memory>

template <Numeric T>
class Optimizer {
public:

    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zero_grad() = 0;

};

template <Numeric T>
class SGD : public Optimizer<T> {
public:
    SGD(const std::vector<TensorS<T>> &params, T learning_rate) 
    : params(params), lr(learning_rate) {}

    void step() override 
    {
        for (auto &p: this->params) {
            for (size_t i = 0; i < p->data.size(); ++i)
                p->data[i] -= this->lr * p->grad[i];
        }
    }

    void zero_grad() override
    {
        for (auto &p: this->params) {
            p->zero_grad();
        }
    }

private:
    std::vector<TensorS<T>> params;
    T lr;
};

#endif