#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "core/tensor_core.hpp"
#include <memory>

#ifdef USE_BLAS
    #include <cblas.h>
    #define BLAS 1
#else
    #define BLAS 0
#endif

#if BLAS
template<Numeric T>
void raw_matmul(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, size_t m, size_t n, size_t p, T beta = 0.0)
{
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            m,      // N
            p,      // P
            n,      // M
            1.0,
            a.data(), n,  // lda = number of columns
            b.data(), p,  // ldb
            beta,
            c.data(), p   // ldc
        );
    } else {
        cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m,      // N
                p,      // P
                n,      // M
                1.0,
                a.data(), n,  // lda = number of columns
                b.data(), p,  // ldb
                beta,
                c.data(), p   // ldc
        );
    }
}
#else
#warning "BLAS DISABLED"
template<Numeric T>
void raw_matmul(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, size_t m, size_t n, size_t p, T beta = 0.0)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum + (beta*c[i*p+j]);
        }
    }
}
#endif

template <typename T>
std::vector<T> transpose(const std::vector<T>& mat, size_t rows, size_t cols)
{
    std::vector<T> result(cols * rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = mat[i * cols + j];
        }
    }

    return result;
}

namespace tensor::ops {

    template<Numeric T>
    TensorS<T> matmul(TensorS<T> A, TensorS<T> B) {
        if (A->shape.size() != 2 || B->shape.size() != 2)
            throw std::runtime_error("matmul only supports 2D tensors");

        size_t m = A->shape[0];
        size_t n = A->shape[1];
        size_t p = B->shape[1];

        if (n != B->shape[0])
            throw std::runtime_error("matmul shapes do not align");

        std::vector<T> out_data(m * p, 0.0);
        raw_matmul(A->data, B->data, out_data, m, n, p);

        auto out = std::make_shared<Tensor<T>>(
                typename Tensor<T>::Shape{m, p},
                out_data,
                A->requires_grad || B->requires_grad,
                std::vector<TensorS<T>>{A, B},
                "MatMulBackward"
        );

        out->grad_fn = [A, B, out, m, n, p]() {
            if (A->requires_grad) {
                auto BT = transpose(B->data, n, p);
                raw_matmul(out->grad, BT, A->grad, m, p, n, T(1));
                for (auto &x: BT) x *= x;
                raw_matmul(out->hess, BT, A->hess, m, p, n, T(1));
            }

            if (B->requires_grad) {
                auto AT = transpose(A->data, m, n);
                raw_matmul(AT, out->grad, B->grad, n, m, p, T(1));
                for (auto &x: AT) x *= x;
                raw_matmul(AT, out->hess, B->hess, n, m, p, T(1));
            }
        };

        return out;
    }
}

#endif