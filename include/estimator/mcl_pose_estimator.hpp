//
// Created by ubuntuvm on 11/24/22.
//

#ifndef VISION_SYSTEM_MCL_POSE_ESTIMATOR_HPP
#define VISION_SYSTEM_MCL_POSE_ESTIMATOR_HPP

#define DT 0.05
#define ALPHA_ROTATION 0.0002
#define ALPHA_TRANSLATION 2
#define NUM_PARTICLES 200
#define RESAMPLE_PARTICLES 100

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "map.hpp"
#include "utils/utils.hpp"
#include "estimator.hpp"
#include "utils/probability_utils.hpp"

struct Particle {
    Eigen::Vector3d x;
    double weight;
};

template <typename T>
class MCLPoseEstimator : public Estimator<T>{
public:
    MCLPoseEstimator() = default;
    explicit MCLPoseEstimator(const std::vector<Landmark> &map);
    ~MCLPoseEstimator();

    virtual void run();
    virtual void setup();

    std::vector<Measurement<T>> measurement_model(const Eigen::Vector<T, 3> &x);

    T sample_measurement_model(
            const Measurement<T> &measurement,
            const Eigen::Vector<T, 3> &x,
            const Landmark &landmark);

    Eigen::Vector3<T> sample_motion_model(
            const ControlInput<T> &u,
            const Eigen::Vector<T, 3> &x);

    T calculate_weight(const std::vector<Measurement<T>> &z,
                            const Eigen::Vector<T, 3> &x,
                            T weight,
                            const std::vector<Landmark> &map);

    std::vector<Particle> monte_carlo_localization(
            const ControlInput<T> &u,
            const std::vector<Measurement<T>> &z);

    std::vector<Particle> low_variance_sampler(const std::vector<Particle> &X);

    std::vector<Particle> get_particle_set();
    Eigen::Vector<T, 3> get_estimated_pose();


private:
    Eigen::Vector<T, 3> x_est_;

    std::vector<Particle> X_; // particle set for filter
    std::vector<Landmark> map_;
};
#endif //VISION_SYSTEM_MCL_POSE_ESTIMATOR_HPP
