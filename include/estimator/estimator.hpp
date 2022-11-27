//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_ESTIMATOR_HPP
#define VISION_SYSTEM_ESTIMATOR_HPP
#include <Eigen/Geometry>
//#include "networking/odometry_sub.hpp"
//#include "networking/state_estimate_pub.hpp"

using namespace Eigen;

template <typename T>
struct ControlInput {
    Translation<T, 2> d_translation; // change in translation;
    Rotation2D<T> d_theta; // change in heading;
//    void set_odometry_input(const odometry_subscribable* msg) {
//        d_translation.x() = msg->dx;
//        d_translation.y() = msg->dy;
//        d_theta.angle() = msg->d_theta;
//    }
};

template <typename T>
struct StateEstimate {
    Translation<T, 2> translation_estimate;
    Rotation2D<T> rotation_estimate;

//    void set_state_estimator_pub_(state_estimate_publishable &data) {
//        data.state_estimate[0] = translation_estimate.x();
//        data.state_estimate[1] = translation_estimate.y();
//        data.state_estimate[2] = rotation_estimate.angle();
//    }
};

template <typename T>
class Measurement {
public:
    Measurement(T range, T bearing, int id) {
        range_ = range;
        bearing_ = bearing;
        id_ = id;
    }

    T get_range() const{
        return range_;
    }

    T get_bearing() const {
        return bearing_;
    }

    int get_element() const {
        return id_;
    }

    void print_measurement() {
        debug(
                "Measurement {Range: " + std::to_string(range_) +
                ", Bearing: " + std::to_string(bearing_) +
                ", Element: " + std::to_string(id_));
    }

private:
    T range_;
    T bearing_;
    int id_;
};

template <typename T>
struct StateEstimatorData {
    StateEstimate<T>* state_estimate;
    ControlInput<T>* control_input;
    std::vector<Measurement<T>>* measurements;
};

template <typename T>
class Estimator {
public:
    virtual void run() = 0;
    virtual void setup() = 0;

    void set_data(StateEstimatorData<T> data) { state_estimator_data_ = data; }

    virtual ~Estimator() = default;
    StateEstimatorData<T> state_estimator_data_;
};

template <typename T>
class EstimatorContainer {
public:
    EstimatorContainer(
            ControlInput<T> *u,
            std::vector<Measurement<T>> *z,
            StateEstimate<T> *estimate
    ) {
        data_.control_input = u;
        data_.measurements = z;
        data_.state_estimate = estimate;
    }

    void run() {
        for (auto estimator : estimators_) {
            estimator->run();
        }
    }

    template<typename EstimatorToAdd>
    void add_estimator() {
        auto *estimator = new EstimatorToAdd();
        estimator->set_data(data_);
        estimator->setup();
        estimators_.push_back(estimator);
    }

    template <typename EstimatorToRemove>
    void removeEstimator() {
        int removed = 0;
        estimators_.erase(
                std::remove_if(estimators_.begin(), estimators_.end(),
                               [&removed](Estimator<T>* e) {
                                   if (dynamic_cast<EstimatorToRemove*>(e)) {
                                       delete e;
                                       removed++;
                                       return true;
                                   } else {
                                       return false;
                                   }
                               }),
                estimators_.end());
    }

    void remove_all_estimators() {
        for (auto estimator : estimators_) {
            delete estimator;
        }
        estimators_.clear();
    }

    ~EstimatorContainer() {
        for (auto estimator : estimators_) {
            delete estimator;
        }
    }
private:
    StateEstimatorData<T> data_;
    std::vector<Estimator<T> *> estimators_;
};

#endif //VISION_SYSTEM_ESTIMATOR_HPP