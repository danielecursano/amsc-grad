#include <iostream>
#include <memory>
#include <cassert>
#include "tensor.hpp"

bool approx(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

int main() {

    {
        auto x = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{1},
                std::vector<double>{1.0},
                true
        );

        auto y = 5.0 * x + pow(x, 2);
        y->backward();
        assert(approx(y->data[0], 6.0));
        assert(approx(x->grad[0], 7.0));  // 5 + 2*1
        assert(approx(x->hess[0], 2.0));
    }

    {
        auto x = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{1},
                std::vector<double>{1.0},
                true
        );
        auto b = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{1},
                std::vector<double>{2.0},
                true
        );

        auto y = x + b;
        y->backward();
        assert(approx(y->data[0], 3.0));

        assert(approx(x->grad[0], 1.0));
        assert(approx(b->grad[0], 1.0));

        assert(approx(x->hess[0], 0.0));
        assert(approx(b->hess[0], 0.0));

    }

    {
        auto x = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{1},
                std::vector<double>{1.0},
                true
        );
        auto b = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{1},
                std::vector<double>{2.0},
                true
        );

        auto y = x * b;
        y->backward();
        assert(approx(y->data[0], 2.0));

        assert(approx(x->grad[0], 2.0));
        assert(approx(b->grad[0], 1.0));

        assert(approx(x->hess[0], 0.0));
        assert(approx(b->hess[0], 0.0));

    }

    std::cout << "All tests passed!\n";
}
