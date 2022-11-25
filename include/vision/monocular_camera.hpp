//
// Created by DSlobodzian on 1/27/2022.
//

#ifndef PARTICLE_FILTER_MONOCULARCAMERA_HPP
#define PARTICLE_FILTER_MONOCULARCAMERA_HPP
#define USE_MATH_DEFINES_


#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "estimator/mcl_pose_estimator.hpp"
#include "map.hpp"
#include "utils/utils.hpp"
#include "camera_config.hpp"




class MonocularCamera {
private:
    cv::VideoCapture cap_;
    cv::Mat frame_;
    int device_id_;
    CameraConfig config_;

public:
    MonocularCamera() = default;
    MonocularCamera(CameraConfig config);
    ~MonocularCamera();

    IntrinsicParameters get_intrinsic_parameters();
    bool open_camera();
    bool read_frame();
    int get_id();

    cv::Mat get_frame();
    void get_frame(cv::Mat& image);
    void draw_rect(cv::Rect rect);
    void draw_crosshair(cv::Rect rect);
};

#endif //PARTICLE_FILTER_MONOCULARCAMERA_HPP
