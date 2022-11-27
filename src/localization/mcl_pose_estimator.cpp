//
// Created by DSlobodzian on 1/6/2022.
//
#include "estimator/mcl_pose_estimator.hpp"

template <typename T>
MCLPoseEstimator<T>::MCLPoseEstimator(const std::vector<Landmark> &map) {
    map_ = map;
}

template <typename T>
void MCLPoseEstimator<T>::setup() {
    Particle p;
    Eigen::Vector3<T> rand_pose;

    p.weight = 1.0 / NUM_PARTICLES;
    X_.assign(NUM_PARTICLES, p);

    x_est_ = Eigen::Vector3<T>::Zero();
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        rand_pose = {
                uniform_random(0.0, 5.0),
                uniform_random(0.0, 5.0),
                uniform_random(0.0, 5.0)
        };
        X_.at(i).x = rand_pose;
    }
}

template <typename T>
void MCLPoseEstimator<T>::run() {
    ControlInput<T> u = this->state_estimator_data_.state_estimate->u;
    std::vector<Measurement<T>> measurements = this->state_estimator_data_.state_estimate->measurements;
    monte_carlo_localization(u, measurements);

    Translation<T, 2> t_est {get_estimated_pose().x(), get_estimated_pose().y()};
    Rotation2D<T> rot_est{get_estimated_pose().z()};
    this->state_estimator_data_.state_estimate->translation_estimate = {t_est, rot_est};
}

// for now assume feature is [range, bearing, element]
template <typename T>
T MCLPoseEstimator<T>::sample_measurement_model(
        const Measurement<T> &measurement,
        const Eigen::Vector3<T> &x,
        const Landmark &landmark) {
    T q = 0;
    if (measurement.get_element() == landmark.tag_id) {
        T range = hypot(landmark.x - x.x(), landmark.y - x.y());
        T bearing = atan2(
                landmark.y - x.y(),
                landmark.x - x.x()
        ) - x.z();
        q = zero_mean_gaussian(measurement.get_range() - range, 0.1) *
            zero_mean_gaussian(measurement.get_bearing() - bearing,0.1);
//        info("probability of meansurement: " + std::to_string(q));
    }
    return q;
}
template <typename T>
Eigen::Vector3<T> MCLPoseEstimator<T>::sample_motion_model(
        ControlInput<T>* u,
        const Eigen::Vector3<T> &x) {
    T noise_dx = u->dx + sample_triangle_distribution(fabs(u->d_translation.x() * ALPHA_TRANSLATION));
    T noise_dy = u->dy + sample_triangle_distribution(fabs(u->d_translation.y() * ALPHA_TRANSLATION));
    T noise_dTheta = u->d_theta + sample_triangle_distribution(fabs(u->d_theta.angle() * ALPHA_TRANSLATION));

    T x_prime = x.x() + noise_dx;
    T y_prime = x.y() + noise_dy;
    T theta_prime = x.z() + noise_dTheta;
    Eigen::Vector3<T> result;
    result << x_prime, y_prime, theta_prime;
    return result;
}

template <typename T>
std::vector<Measurement<T>> MCLPoseEstimator<T>::measurement_model(const Eigen::Vector3<T> &x) {
    std::vector<Measurement<T>> z_vec;
    for (Landmark landmark : map_) {
        T range = hypot(
                landmark.x - x.x(),
                landmark.y - x.y());
        T bearing = atan2(
                landmark.y - x.y(),
                landmark.x - x.x()) - x.z();
        Measurement<T> z(range, bearing, landmark.tag_id);
        z_vec.emplace_back(z);
    }
    return z_vec;
}
template <typename T>
T MCLPoseEstimator<T>::calculate_weight(
        const std::vector<Measurement<T>> &z,
        const Eigen::Vector3<T> &x,
        T weight,
        const std::vector<Landmark> &map) {
    for (Landmark landmark : map) {
        for (auto &i : z) {
            if (i.get_id() == landmark.tag_id) {
                weight = weight * sample_measurement_model(i, x, landmark);
                break;
            }
        }
    }
//    info("Weight :" + std::to_string(weight));
    return weight;
}

template <typename T>
std::vector<Particle> MCLPoseEstimator<T>::low_variance_sampler(const std::vector<Particle> &X) {
    std::vector<Particle> X_bar;
    T r = random(0.0, 1.0 / NUM_PARTICLES);
    T c = X.at(0).weight;
    int i = 0;
    for (int particle = 1; particle <= NUM_PARTICLES; ++particle) {
        T u = r + (T)(particle - 1) * (1.0 / NUM_PARTICLES);
        while (u > c) {
            i = i + 1;
            c = c + X.at(i).weight;
        }
        X_bar.emplace_back(X.at(i));
    }
    return X_bar;
}

template <typename T>
std::vector<Particle> MCLPoseEstimator<T>::monte_carlo_localization(
        ControlInput<T>* u,
        std::vector<Measurement<T>>* z) {
    std::vector<Particle> X_bar;
    T sum = 0;
    Eigen::MatrixX<T> x_set(3, X_.size());
    for (int particle = 0; particle < NUM_PARTICLES; ++particle) {
        Eigen::Vector3<T> x = sample_motion_model(u, X_.at(particle).x);
        T weight = calculate_weight(z, x, X_.at(particle).weight, map_);
        sum += weight;
        Particle p;
        p.x = x;
        p.weight = weight;
        X_bar.emplace_back(p);
    }
    Eigen::VectorX<T> weights(NUM_PARTICLES);
    //Normalize the weights
    for (int i = 0; i < X_.size(); ++i) {
//        std::cout << "original weight is" << X_bar.at(i).weight << "\n";
//        std::cout << "normalized weight is" << X_bar.at(i).weight/sum << "\n";
        X_bar.at(i).weight = X_bar.at(i).weight / sum;
        x_set.col(i) = X_bar.at(i).x;
        weights(i,0) = X_bar.at(i).weight;
    }

    x_est_ = x_set * weights;

    T effective_particles = 1.0 / (weights.transpose() * weights)(0,0);
    if (effective_particles < RESAMPLE_PARTICLES) {
        X_ = low_variance_sampler(X_bar);
    } else {
        X_ = X_bar;
    }
    return X_;
}

template <typename T>
std::vector<Particle> MCLPoseEstimator<T>::get_particle_set() {
    return X_;
}

template <typename T>
Eigen::Vector3<T> MCLPoseEstimator<T>::get_estimated_pose() {
    return x_est_;
}