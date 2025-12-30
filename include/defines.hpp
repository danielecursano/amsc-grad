#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <concepts>

template <typename T>
concept Numeric = std::floating_point<T>;

#endif // DEFINES_HPP