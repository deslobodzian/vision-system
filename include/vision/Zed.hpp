//
// Created by DSlobodzian on 11/5/2021.
//
#pragma once

#include <sl/Camera.hpp>
#include <thread>
#include "utils.hpp"
#include <chrono>
#include <Eigen/Dense>
#include "map.hpp"
#include "localization/particle_filter.hpp"

#define CAM_TO_ROBOT_X -0.361696
#define CAM_TO_ROBOT_Y -0.00889
#define CAM_TO_CATAPULT_Y -0.1651

using namespace sl;

typedef struct {
    Timestamp timestamp;
    Mat left_image;
    Pose camera_pose;
    Mat depth_map;
    Mat point_cloud;
} ZedMeasurements;

class Zed {

private:
     Camera zed_;
     ZedMeasurements measurements_;

     InitParameters init_params_;
     RuntimeParameters runtime_params_;
     ObjectDetectionParameters detection_params_;
     ObjectDetectionRuntimeParameters objectTracker_params_rt_;

     CalibrationParameters calibration_params_;
     Transform cam_to_robot_;

     float left_offset_to_center_;
     bool successful_grab();

public:
    Zed();
    ~Zed();

    bool open_camera();
    bool enable_tracking();
    bool enable_tracking(Eigen::Vector3d init_pose);
    bool enable_object_detection();
    void input_custom_objects(std::vector<sl::CustomBoxObjectData> objects_in);

    void fetch_measurements();

    // gets the last fetched depth map.
    sl::Mat get_depth_map() const;

    // gets the last fetched point cloud.
    sl::Mat get_point_cloud() const;
    sl::float3 get_position_from_pixel(int x, int y);
    sl::float3 get_position_from_pixel(const cv::Point &p);
    double get_distance_from_point(const sl::float3& p);

    sl::Transform get_calibration_stereo_transform() const;

    // gets the last fetched left image.
    sl::Mat get_left_image() const;

    // gets the last fetched camera pose.
    Pose get_camera_pose() const;

    // gets the timestamp from the last fetched measurement information.
    Timestamp get_measurement_timestamp() const;

    void close();

};
