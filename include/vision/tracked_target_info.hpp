//
// Created by DSlobodzian on 4/21/2022.
//

#ifndef VISION_SYSTEM_TARGET_INFO_HPP
#define VISION_SYSTEM_TARGET_INFO_HPP

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "apriltag_detector.hpp"
#include <Eigen/Dense>
#include "map.hpp"

class TrackedTargetInfo {
private:
    std::string target_identity_;
    cv::Point target_center_;
    Corners corners_;
    double x_;
    double y_;
    double z_;

    double vx_;
    double vy_;
    double vz_;

public:
    TrackedTargetInfo();
    ~TrackedTargetInfo();

    const double get_x();
    const double get_y();
    const double get_z();

    const double get_vx();
    const double get_vy();
    const double get_vz();

    double get_distance(double x_offset, double y_offset, double z_offset);
    double get_distance(sl::Transform offset);
    double get_yaw_angle();
    double get_pitch_angle();
    std::string to_packet();
};

#endif //VISION_SYSTEM_TARGET_INFO_HPP
