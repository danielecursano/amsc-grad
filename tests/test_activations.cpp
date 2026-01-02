#include <iostream>
#include <memory>
#include <cassert>
#include "tensor.hpp"

bool approx(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

int main() {
    using namespace tensor::ops;
    using T = float;

    auto x = std::make_shared<Tensor<T>>(
            Tensor<T>::Shape{2, 2},
            std::vector<T>{1.0, 0.0, 3.0, 4.0},
            true
    );

    {
        auto y = relu(x);

        y->backward();

        assert(approx(y->data[0], 1.0));
        assert(approx(y->data[1], 0.0));
        assert(approx(y->data[2], 3.0));
        assert(approx(y->data[3], 4.0));

        assert(approx(x->grad[0], 1.0));
        assert(approx(x->grad[1], 0.0));
        assert(approx(x->grad[2], 1.0));
        assert(approx(x->grad[3], 1.0));

        std::cout << "Relu test passed" << std::endl;
    }

    {
        x->zero_grad();
        auto y = tanh(x);
        y->backward();
        
    }

}