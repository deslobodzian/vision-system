//
// Created by DSlobodzian on 11/5/2021.
//
#pragma once

#include <sl/Camera.hpp>
#include <thread>
#include "utils/utils.hpp"
#include <chrono>
#include <Eigen/Dense>
#include "map.hpp"
#include "estimator/mcl_pose_estimator.hpp"
#include "camera.hpp"

using namespace sl;

typedef struct {
    Timestamp timestamp;
    Mat left_image;
    Pose camera_pose;
    Mat depth_map;
    Mat point_cloud;
} ZedMeasurements;

class Zed : public GenericCamera {


public:
    Zed(const zed_config &config);
    ~Zed();

    CAMERA_TYPE get_camera_type() const override {
        return ZED;
    };

    int open_camera();
    int enable_tracking();
    int enable_tracking(const Eigen::Vector3f &init_pose);

    void fetch_measurements();

    // gets the last fetched depth map.
    sl::Mat get_depth_map() const;

    // gets the last fetched point cloud.
    sl::Mat get_point_cloud() const;
    sl::float3 get_position_from_pixel(int x, int y) const;
    sl::float3 get_position_from_pixel(const cv::Point &p);
    float get_distance_from_point(const sl::float3& p);

    sl::Transform get_calibration_stereo_transform() const;

    // gets the last fetched left image.
    sl::Mat get_left_image() const;

    // gets the last fetched camera pose.
    Pose get_camera_pose() const;

    // gets the timestamp from the last fetched measurement information.
    Timestamp get_measurement_timestamp() const;

    void close();
private:
    Camera zed_;
    ZedMeasurements measurements_;
    InitParameters init_params_;
    CalibrationParameters calibration_params_;

    float left_offset_to_center_{};
    bool successful_grab();
};
