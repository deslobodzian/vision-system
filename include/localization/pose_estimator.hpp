//
// Created by DSlobodzian on 1/27/2022.
//

#pragma once

#include <iostream>
#include <thread>
#include "particle_filter.hpp"
#include "map.hpp"
#include "vision/Zed.hpp"
#include "vision/apriltag_manager.hpp"
#include "vision/monocular_camera.hpp"


class PoseEstimator {
private:
    Eigen::Vector3d init_pose_;
    std::vector<Measurement> z_;
    ControlInput u_;

    ParticleFilter filter_;
    AprilTagManager april_tag_manager_;
public:
    PoseEstimator() = default;
    ~PoseEstimator();

    void add_to_measurement_vector(const std::vector<TrackedTargetInfo> &detected_targets);
    void fetch_current_measurements();

    [[noreturn]] void estimate_pose();
    void init();
};
