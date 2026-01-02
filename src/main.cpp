#include <iostream>
#include "tensor.hpp"
#include <cmath>

int main() {
    using T = float;

    set_seed(32);

    auto real_solution = [](auto x, auto y) {
        return x*x - y*y;
    };

    size_t N_collocation = 400;
    size_t N_boundaries = 120;

    auto x = uniform<T>({N_collocation, 2}, -1.f, 1.f, true);

    auto y_true = zeros<T>({N_collocation, 1});
    for (size_t i = 0; i < N_collocation; ++i)
        y_true->data[i] = real_solution(x->data[i*2], x->data[i*2+1]);


    size_t Nb_side = N_boundaries / 4;
    auto x_boundaries = uniform<T>({N_boundaries, 2}, -1.f, 1.f, false);

    for (size_t i = 0; i < N_boundaries; ++i) {
        if (i < Nb_side) {
            x_boundaries->data[i*2] = -1.f;
        }
        else if (i < 2 * Nb_side) {
            x_boundaries->data[i*2] = +1.f;
        }
        else if (i < 3 * Nb_side) {
            x_boundaries->data[i*2 + 1] = -1.f;
        }
        else {
            x_boundaries->data[i*2 + 1] = +1.f;
        }
    }

    auto boundary_target = zeros<T>({N_boundaries, 1}, false);
    for (size_t i = 0; i < N_boundaries; ++i) {
        boundary_target->data[i] = real_solution(
                x_boundaries->data[i*2],
                x_boundaries->data[i*2+1]
        );
    }

    auto W1 = normal<T>({2, 20}, 0., 0.1, true);
    auto W2 = normal<T>({20, 20}, 0., 0.1, true);
    auto W3 = normal<T>({20, 20}, 0., 0.1, true);
    auto W4 = normal<T>({20, 20}, 0., 0.1, true);
    auto W5 = normal<T>({20, 1}, 0, 0.1, true);
    auto B1 = zeros<T>({1, 20}, true);
    auto B2 = zeros<T>({1, 20}, true);
    auto B3 = zeros<T>({1, 20}, true);
    auto B4 = zeros<T>({1, 20}, true);
    auto B5 = zeros<T>({1, 1}, true);

    int epochs = 1000;
    T lr = 2e-4;

    auto model = [&W1, &W2, &W3, &B1, &B2, &B3, &W4, &W5, &B4, &B5](auto x) {
        auto h =  tanh(broadcast_add(matmul(x, W1), B1));
        auto w =  tanh(broadcast_add(matmul(h, W2), B2));
        auto z =  tanh(broadcast_add(matmul(w, W3), B3));
        auto y =  tanh(broadcast_add(matmul(z, W4), B4));
        return broadcast_add(matmul(y, W5), B5);
    };

    auto mse_loss = [](auto pred, auto target) {
        return mean(pow(pred + (-1.f)*target, 2));
    };

    auto optim = Adam<float>({
                                     AdamVariable(W1, true),
                                     AdamVariable(W2, true),
                                     AdamVariable(B1, true),
                                     AdamVariable(B2, true),
                                     AdamVariable(W3, true),
                                     AdamVariable(B3, true),
                                     AdamVariable(W4, true),
                                     AdamVariable(B4, true),
                                     AdamVariable(W5, true),
                                     AdamVariable(B5, true),
                             }, lr, 0.9, 0.999, 1e-8, 1e-3);

    T lambda_pde = 1.0f;
    T lambda_boundary = 10.0f;

    std::ofstream history("history.csv");
    if (!history.is_open()) {
        throw std::runtime_error("Failed to open output file");
    }
    std::cout << "history,pde_loss,boundary_loss,total_loss\n";
    history << "history,pde_loss,boundary_loss,total_loss\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optim.zero_grad();
        x->zero_grad();
        // Forward pass
        auto pred = model(x);
        pred->backward();

        auto laplacian = zeros<T>({N_collocation, 1}, false);
        for (size_t i = 0; i < N_collocation; ++i)
            laplacian->data[i] = x->hess[i*2] + x->hess[i*2+1];

        auto pde_loss = mean(pow(laplacian, 2));

        // Boundary loss
        auto pred_bound = model(x_boundaries);
        auto boundary_loss = mse_loss(pred_bound, boundary_target);

        // Total loss
        auto total_loss = lambda_pde * pde_loss + lambda_boundary * boundary_loss;

        // Backprop
        optim.zero_grad();
        total_loss->backward();
        optim.step();

        // Logging
        if (epoch % 1 == 0) {
            std::cout << epoch << ","
                      << pde_loss->data[0] << ","
                      << boundary_loss->data[0] << ","
                      << total_loss->data[0] << std::endl;
            history << epoch << ","
                      << pde_loss->data[0] << ","
                      << boundary_loss->data[0] << ","
                      << total_loss->data[0] << std::endl;
        }

    }

    history.close();

    size_t n = 100;
    size_t N = n*n;

    double h = 2.0 / (n-1);

    auto validation_points = zeros<T>({N, 2});
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) {
        T x = -1.0 + i * h;
        for (size_t j = 0; j < n; ++j) {
            T y = -1.0 + j * h;
            validation_points->data[idx*2] = x;
            validation_points->data[idx*2+1] = y;
            ++idx;
        }
    }

    std::ofstream file("output.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file");
    }
    file << "x,y,u,true_u\n";
    auto output = model(validation_points);
    for (size_t i = 0; i < N; ++i) {
        file << validation_points->data[i*2] << ","
        << validation_points->data[i*2 + 1] << ","
        << output->data[i] << ","
        << real_solution(validation_points->data[i*2], validation_points->data[i*2+1]) << "\n";
    }
    file << std::endl;
    file.close();

    return 0;
}