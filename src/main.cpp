#include <iostream>
#include "tensor.hpp"
#include <cmath>
#include "extra/GetPot.hpp"

/**
 * Demonstrates a physics-informed neural network to solve a 2d Laplace problem.
 *
 * This example uses a simple feedforward neural network implemented with the tensor library.
 * It approximates the solution to a 2d Laplace problem on a domain [-1, 1] x [-1, 1] with Dirichlet
 * boundary conditions.
 *
 * The code workflow is as follows:
 * 1. Sets the random seed for reproducibility.
 * 2. Defines the exact solution (x^2 - y^2 = 0).
 * 3. Creates the training dataset of Nc collocation points and Nb boundary points.
 * 4. Initializes the neural network parameters and define the forward model.
 * 5. Defines the mean squared error loss and PDE loss.
 * 6. Uses the Adam optimizer to train the network.
 * 7. Generates a grid of points for validation and post-processing data for the python notebook.
 */
int main(int argc, char* argv[]) {
    using T = double;
    using namespace tensor::ops;

    GetPot parser("pinn_config.dat");
    GetPot cmd(argc, argv);

    // Size of hidden layers in the nn
    size_t hidden_size = parser("hidden_size", 20);

    // Number of interior points and boundary points
    size_t N_collocation = parser("N_collocation", 400);
    size_t N_boundaries = parser("N_boundaries", 120);

    // Coefficients for PDE and boundary loss in the total loss formula:
    // L_tot = lambda_pde * PDE_loss + lambda_boundary * B_loss
    T lambda_pde = parser("lambda_pde", 1.0f);
    T lambda_boundary = parser("lambda_boundary", 10.0f);

    // Training parameters
    int epochs = cmd("--epochs", parser("epochs", 1000));
    T lr = cmd("--lr", parser("learning_rate", 2e-4));

    bool verbose = cmd.search("--verbose");
    int OUTPUT_INTERVAL = verbose ? 1 : epochs / 10;

    std::cout << "========================================\n";
    std::cout << "Running 2D Laplace PINN problem\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Collocation points: " << N_collocation << "\n";
    std::cout << "Boundary points: " << N_boundaries << "\n";
    std::cout << "Learning rate: " << lr << "\n";
    std::cout << "Loss weights: lambda_pde = " << lambda_pde
              << ", lambda_boundary = " << lambda_boundary << "\n";
    std::cout << "Training epochs: " << epochs << "\n";
    std::cout << "========================================\n";

    tensor::set_seed(32);

    // Define the exact solution of the PDE: u(x, y) = x^2 - y^2
    auto real_solution = [](auto x, auto y) {
        return x*x - y*y;
    };

    // Dataset
    // Collection points uniformly sampled in [-1, 1] x [-1, 1]
    auto x = tensor::uniform<T>({N_collocation, 2}, -1.f, 1.f, true);

    size_t Nb_side = N_boundaries / 4;
    auto x_boundaries = tensor::uniform<T>({N_boundaries, 2}, -1.f, 1.f, false);
    x_boundaries->metadata.name = "Boundary points";

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

    auto boundary_target = tensor::zeros<T>({N_boundaries, 1}, false);
    for (size_t i = 0; i < N_boundaries; ++i) {
        boundary_target->data[i] = real_solution(
                x_boundaries->data[i*2],
                x_boundaries->data[i*2+1]
        );
    }

    // Neural network
    auto linear1 = tensor::nn::Linear<T>(2, hidden_size, 0.1);
    auto linear2 = tensor::nn::Linear<T>(hidden_size, hidden_size, 0.1);
    auto linear3 = tensor::nn::Linear<T>(hidden_size, hidden_size, 0.1);
    auto linear4 = tensor::nn::Linear<T>(hidden_size, hidden_size, 0.1);
    auto linear5 = tensor::nn::Linear<T>(hidden_size, 1, 0.1);

    // Forward model
    auto model = [&linear1, &linear2, &linear3, &linear4, &linear5](auto x) {
        return linear5(tanh(linear4(tanh(linear3(tanh(linear2(tanh(linear1(x)))))))));
    };

    // Lambda function to compute MSE loss
    auto mse_loss = [](auto pred, auto target) {
        return mean(pow(pred + (-1.)*target, 2));
    };

    // Adam parameters
    T beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 1e-3;

    auto optim = tensor::optim::Adam<T>({
            {linear1.getParams()[0], true},
            {linear1.getParams()[1], true},
            {linear2.getParams()[0], true},
            {linear2.getParams()[1], true},
            {linear3.getParams()[0], true},
            {linear3.getParams()[1], true},
            {linear4.getParams()[0], true},
            {linear4.getParams()[1], true},
            {linear5.getParams()[0], true},
            {linear5.getParams()[1], true},
    }, lr, beta1, beta2, eps, weight_decay);

    // File where to store the history of the training
    std::ofstream history("history.csv");
    if (!history.is_open()) {
        throw std::runtime_error("Failed to open output file");
    }
    history << "history,pde_loss,boundary_loss,total_loss\n";

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optim.zero_grad();
        x->zero_grad();

        auto perm = tensor::random_perm(N_collocation);
        x->permute_rows(perm);   // Permuting the rows of the train dataset

        // Forward pass: computes u'(x)
        auto pred = model(x);
        pred->backward();

        // Computes PDE_loss as: d^2 u' / dx^2 + d^2 u' / dy^2
        auto laplacian = tensor::zeros<T>({N_collocation, 1}, false);
        for (size_t i = 0; i < N_collocation; ++i)
            laplacian->data[i] = x->hess[i*2] + x->hess[i*2+1];

        auto pde_loss = mean(pow(laplacian, 2));
        pde_loss->metadata.name = "pde_loss";
        
        // Boundary loss
        
        auto perm_bound = tensor::random_perm(N_boundaries); // Permuting the rows of the train dataset
        x_boundaries->permute_rows(perm_bound);
        boundary_target->permute_rows(perm_bound);

        auto pred_bound = model(x_boundaries);
        auto boundary_loss = mse_loss(pred_bound, boundary_target);

        // Total loss
        auto total_loss = lambda_pde * pde_loss + lambda_boundary * boundary_loss;
        total_loss->metadata.name = "Total loss";

        // Backpropagation and parameter update
        optim.zero_grad();
        total_loss->backward();
        optim.step();

        // Logging
        if (epoch % OUTPUT_INTERVAL == 0) {
            std::cout << "Epoch: " << epoch << ", PDE loss: "
                      << pde_loss->data[0] << ", Data loss: "
                      << boundary_loss->data[0] << ", Total loss: "
                      << total_loss->data[0] << std::endl;
        }

        history << epoch << ","
                << pde_loss->data[0] << ","
                << boundary_loss->data[0] << ","
                << total_loss->data[0] << std::endl;

    }

    history.close();

    // Validation step
    size_t n = 100;
    size_t N = n*n;

    double h = 2.0 / (n-1);

    auto validation_points = tensor::zeros<T>({N, 2});
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