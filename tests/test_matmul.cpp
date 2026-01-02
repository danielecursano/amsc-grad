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
            std::vector<T>{1.0, 2.0, 3.0, 4.0},
            true
    );
    auto y = std::make_shared<Tensor<T>>(
            Tensor<T>::Shape{2, 2},
            std::vector<T>{5.0, 6.0, 7.0, 8.0},
            true
    );

    auto c = matmul(x, y);
    c->backward();

    std::vector<T> expected_forward{19, 22, 43, 50};
    for (size_t i = 0; i < 4; ++i) {
        assert(approx(c->data[i], expected_forward[i]));
    }

    std::vector<T> expected_grad_A{11, 15, 11, 15};
    for (size_t i = 0; i < 4; ++i) {
        assert(approx(x->grad[i], expected_grad_A[i]));
    }

    std::vector<T> expected_grad_B{4, 4, 6, 6};
    for (size_t i = 0; i < 4; ++i) {
        assert(approx(y->grad[i], expected_grad_B[i]));
    }

    std::cout << "Matmul tests passed!\n";

    return 0;

}