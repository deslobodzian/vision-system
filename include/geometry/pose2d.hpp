//
// Created by ubuntuvm on 11/14/22.
//

#ifndef VISION_SYSTEM_POSE2D_HPP
#define VISION_SYSTEM_POSE2D_HPP

#include <sophus/geometry.hpp>
#include <Eigen/Dense>

class Pose2d {
private:
    Sophus::SE2d pose_;
public:
    Pose2d() = default;

    Pose2d(double x, double y, double angle) : pose_(angle, Eigen::Vector2d(x, y)) {};

    Pose2d(Eigen::Translation2d &translation, Eigen::Rotation2Dd &rot) : pose_(rot.matrix(), translation.translation()){};
    Pose2d(Eigen::Translation2d &translation, Sophus::SO2d &rot) : pose_(rot, translation.translation()) {};
};

#endif //VISION_SYSTEM_POSE2D_HPP
