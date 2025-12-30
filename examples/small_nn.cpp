#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    using T = float;

    // --- Dataset ---
    // 2 features, 4 examples
    std::vector<std::vector<T>> X_data = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}
    };
    std::vector<T> y_data = {
            0.0f, 2.0f, 1.0f, 3.0f
    };

    // Flatten input for tensor
    std::vector<T> X_flat;
    for (auto &row : X_data) X_flat.insert(X_flat.end(), row.begin(), row.end());

    auto input = std::make_shared<Tensor<T>>(Tensor<T>::Shape{4, 2}, X_flat);
    auto target = std::make_shared<Tensor<T>>(Tensor<T>::Shape{4, 1}, y_data);

    // --- Parameters ---
    auto w = normal<T>({2, 1}, 0.0f, 1.0f, true);
    auto b = zeros<T>({1, 1}, true);

    float lr = 0.1f;

    auto optim = Adam<float>({AdamVariable(w, true), AdamVariable(b, false)}, lr, 0.1, 0.1, 0.1, 0.1);
    //auto optim = SGD({w, b}, lr);

    auto model = [w, b](auto input) {
        return relu(matmul(input, w)+b);
    };

    auto loss_fn = [](auto pred, auto target) {
        return pow(pred + (-1.0f* target), 2);
    };

    // --- Training loop ---
    for (int epoch = 0; epoch < 100; ++epoch) {
        auto pred = model(input);

        // Mean squared error: (pred - target)^2
        auto loss = loss_fn(pred, target);
        T loss_val = 0.0;
        for (auto v : loss->data) loss_val += v;
        loss_val /= 4.0f; // batch size

        // Backward
        loss->backward();
        
        optim.step();
        optim.zero_grad();

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << " loss: " << loss_val << std::endl;
    }

    // --- Prediction ---
    auto pred = matmul(input, w) + b;
    std::cout << "Predictions:\n";
    for (auto v : pred->data) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
