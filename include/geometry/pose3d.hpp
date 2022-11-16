//
// Created by ubuntuvm on 11/14/22.
//

#ifndef VISION_SYSTEM_POSE3D_HPP
#define VISION_SYSTEM_POSE3D_HPP
#include <sophus/geometry.hpp>
#include <Eigen/Dense>

class Pose3d {
private:
    Sophus::SE3d pose_;
public:
    Pose3d() = default;
    Pose3d(Eigen::Translation3d &translation, Sophus::SO3d &rot) : pose_(rot, translation.translation()) {}
    Pose3d(double x, double y, double z, double roll, double pitch, double yaw) {
        Sophus::SO3d roll_mat = Sophus::SO3d::rotX(roll);
        Sophus::SO3d pitch_mat = Sophus::SO3d::rotY(pitch);
        Sophus::SO3d yaw_mat = Sophus::SO3d::rotZ(yaw);
        Sophus::SO3d rotation_mat = roll_mat * pitch_mat * yaw_mat;
        Eigen::Translation3d translation(x, y, z);
        pose_ = Sophus::SE3d(rotation_mat, translation.translation());
    }

};

#endif //VISION_SYSTEM_POSE3D_HPP
