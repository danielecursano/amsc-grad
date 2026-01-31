#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "core/tensor_core.hpp"
#include "utils/tensor_utils.hpp"
#include "ops/matmul.hpp"
#include "ops/arithmetic.hpp"


namespace tensor::nn {

template <Numeric T>
class Layer {
    public:
        virtual ~Layer() = default;
        virtual std::vector<TensorS<T>> getParams() const = 0;
        virtual TensorS<T> operator()(const TensorS<T>) const = 0;
};

template <Numeric T>
class Linear: public Layer<T> {

    public:

        Linear(Tensor<T>::shape_type input_dims, Tensor<T>::shape_type dims, T std = 1.0) : 
        W(tensor::normal<T>({input_dims, dims}, T(0), std, true)),
        b(tensor::zeros<T>({1, dims}, true)) {}

        std::vector<TensorS<T>> getParams() const override 
        {
            return {W, b};
        }

        TensorS<T> operator()(const TensorS<T> x) const override 
        {
            return tensor::ops::broadcast_add(tensor::ops::matmul(x, W), b);
        }

    private:
        TensorS<T> W, b;
};

}

#endif