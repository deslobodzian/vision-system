//
// Created by DSlobodzian on 1/28/2022.
//

#include "localization/pose_estimator.hpp"



PoseEstimator::~PoseEstimator(){
}

void PoseEstimator::fetch_current_measurements() {
    z_.clear(); // clear measurement vector to make sure previous targets are not used.
    add_to_measurement_vector(april_tag_manager_.get_zed_targets());
    add_to_measurement_vector(april_tag_manager_.get_monocular_targets());
}

void PoseEstimator::add_to_measurement_vector(const std::vector<TrackedTargetInfo> &detected_targets) {
    for (TrackedTargetInfo t : detected_targets) {
        z_.emplace_back(Measurement(t.get_distance(), t.get_yaw_angle(), t.get_id()));
    }
}

void PoseEstimator::estimate_pose() {
    for (;;) {
        auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        fetch_current_measurements();
        filter_.monte_carlo_localization(u_, z_);
        std::this_thread::sleep_until(x);
    }
}


void PoseEstimator::init() {

}



