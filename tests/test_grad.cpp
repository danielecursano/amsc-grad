#include <iostream>
#include <memory>
#include <cassert>
#include "tensor.hpp"

bool approx(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}

int main() {
    using namespace tensor;
    using namespace tensor::ops;
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

    {
        auto x = std::make_shared<Tensor<double>>(
                Tensor<double>::Shape{5},
                std::vector<double>{1.0, 1.0, 2.0, 3.0, 5.0},
                true
        );

        auto y = mean(x);
        y->backward();
        assert(approx(y->data[0], (12. / 5.)));
    }

    {
        // example from the slides of NAML at PoliMi
        // v = -2x1 + 3x2 + 0.5
        // h = tanh v
        // y = 2 * h - 1
        auto x1 = zeros<float>({1}, true);
        x1->metadata.name = "x1";
        auto x2 = ones<float>({1}, true);
        x2->metadata.name = "x2";
        auto c1 = zeros<float>({1});
        auto c2 = zeros<float>({1});
        c1->data[0] = 0.5;
        x1->data[0] = 2.f;
        c2->data[0] = -1.f;
        auto v = -2.f * x1 + 3.f * x2 + c1;
        auto h = tanh(v);
        auto y = 2.f * h + c2;
        y->metadata.name = "y";
        y->backward();
    }

    std::cout << "All tests passed!\n";
}
