#ifndef MODEL_HPP
#define MODEL_HPP

#include "core/tensor_core.hpp"

namespace tensor::nn {

    /**
     * @brief Abstract interface of a neural network.
     */
    template <Numeric T>
    struct Model {

        virtual ~Model() = default;
        
        virtual TensorS<T> operator()(const TensorS<T>& input) const = 0;

        virtual std::vector<TensorS<T>> getParams() const = 0;

    };

}

#endif
