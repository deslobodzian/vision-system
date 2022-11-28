//
// Created by ubuntuvm on 11/24/22.
//

#ifndef VISION_SYSTEM_PROBABILITY_UTIL_HPP
#define VISION_SYSTEM_PROBABILITY_UTIL_HPP
#include <cmath>
#include <random>

template <typename T>
inline T random(T min, T max) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> dist(min, max);
    return dist(mt);
}

template <typename T>
inline T uniform_random(T center, T range) {
    return random(center - range, center + range);
}

template <typename T>
inline T sample_triangle_distribution(T b) {
    return (sqrt(6.0)/2.0) * (random(-b, b) + random(-b, b));
}

template <typename T>
inline T zero_mean_gaussian(T x, T sigma) {
    return (1.0 / sqrt(2.0 * M_PI * pow(sigma, 2))) * exp(-pow(x, 2) / (2.0 * pow(sigma, 2)));
}

#endif //VISION_SYSTEM_PROBABILITY_UTIL_HPP
