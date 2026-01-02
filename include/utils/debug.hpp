#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <fstream>
#include <ostream>
#include <string>
#include <iostream>
#include <sstream>

#include "core/tensor_core.hpp"

/**
 * @brief Writes a representation of a tensor to an output stream.
 *
 * The representation includes the shape of the tensor and its gradient function.
 *
 * @param os Output stream
 * @param t Input tensor
 * @return Reference to the output stream os
 */
template <Numeric T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) 
{
    os << "Tensor(shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        os << t.shape[i];
        if (i+1 < t.shape.size()) os << ", ";
    }
    os << "], grad_fn=<" << t.metadata << ">)" << std::endl;
    return os;
}

/**
 * Writes the computation graph of the backpropagation path to a Graphviz .dot file.
 *
 * @tparam T
 * @param graph Vector of shared pointers to tensors representating the nodes of the graph
 */
template <Numeric T>
void print_graph(const std::vector<TensorS<T>>& graph)
{
    std::ofstream file("graph.dot");
    if (!file.is_open()) {
        std::cerr << "Failed to open file graph.dot\n";
        return;
    }

    file << "digraph ComputationGraph {\n";

    for (auto& node : graph) {
        std::ostringstream label;
        label << *node;

        std::string raw = label.str();
        std::string escaped;

        for (char c : raw) {
            if (c == '\n') escaped += "\\n";
            else if (c == '"') escaped += "\\\"";
            else escaped += c;
        }

        file << "  \"node_" << node.get()
             << "\" [label=\"" << escaped << "\"];\n";

        for (auto& p : node->prev) {
            file << "  \"node_" << p.get()
                 << "\" -> \"node_" << node.get() << "\";\n";
        }
    }

    file << "}\n";
}

#endif