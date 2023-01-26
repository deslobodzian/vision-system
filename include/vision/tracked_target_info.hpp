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
    int id_;
    float x_;
    float y_;
    float z_;
    float vx_;
    float vy_;
    float vz_;

public:
    TrackedTargetInfo();
    TrackedTargetInfo(float x, float y, float z, int id);
    TrackedTargetInfo(sl::Pose &pose, int id);
    TrackedTargetInfo(const sl::ObjectData& object_data);
    ~TrackedTargetInfo();

    float get_x() const;
    float get_y() const;
    float get_z() const;
    int get_id() const;

    double get_distance();
    double get_distance(double x_offset, double y_offset, double z_offset);
    double get_distance(const sl::Transform& offset);

    double get_yaw_angle();
    double get_pitch_angle();

    std::vector<float> get_vec() const;

};

#endif //VISION_SYSTEM_TARGET_INFO_HPP
