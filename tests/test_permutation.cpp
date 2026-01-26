#include <iostream>
#include <memory>
#include <cassert>
#include "tensor.hpp"

int main() {

    auto id = tensor::eye<float>({3, 3});
    auto perm = tensor::random_perm(3);

    id->permute_rows(perm);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = id->data[i * 3 + j];

            if (j == perm[i]) {
                assert(v == 1.0f);
            } else {
                assert(v == 0.0f);
            }
        }
    }
    std::cout << "Row permutation test passed ✅\n";
}