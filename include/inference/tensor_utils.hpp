#ifndef VISION_SYSTEM_TENSOR_UTILS_HPP
#define VISION_SYSTEM_TENSOR_UTILS_HPP

#include <vector>
#include <string>
#include <numeric>
#include <type_traits>
#include <iterator>

template <typename Container>
auto vector_product(const Container& c) -> decltype(*std::begin(c)) {
    using T = decltype(*std::begin(c));
    static_assert(std::is_arithmetic<T>::value, "vectorProduct requires numeric container elements.");
    return std::accumulate(std::begin(c), std::end(c), static_cast<T>(1), std::multiplies<T>());
}

template <typename Container>
std::string tensor_shape_to_string(const Container& shape) {
    std::string tensor_shape = "Tensor shape is [";
    for (auto it = std::begin(shape); it != std::end(shape); ++it) {
        tensor_shape += std::to_string(*it);
        if (std::next(it) != std::end(shape)) {
            tensor_shape += ", ";
        }
    }
    tensor_shape += "]";
    return tensor_shape;
}

#endif /* VISION_SYSTEM_TENSOR_UTILS_HPP */