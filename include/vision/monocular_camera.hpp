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
#include "map.hpp"
#include "utils/utils.hpp"
#include "camera_config.hpp"
#include "camera.hpp"


template <typename T>
class MonocularCamera : GenericCamera {
private:
    cv::VideoCapture cap_;
    cv::Mat frame_;
    int device_id_ = 0;
    CameraConfig<T> config_;

public:
    MonocularCamera() = default;
    explicit MonocularCamera(CameraConfig<T> config);
    ~MonocularCamera();

    IntrinsicParameters<T> get_intrinsic_parameters();
    int open_camera() override;
    void fetch_measurements() override;
    int get_id() const;

    CAMERA_TYPE get_camera_type() const override{
        return MONOCULAR;
    }

    cv::Mat get_frame();
    void get_frame(cv::Mat& image);
    void draw_rect(const cv::Rect &rect);
    void draw_crosshair(const cv::Rect &rect);
};

#endif //PARTICLE_FILTER_MONOCULARCAMERA_HPP
