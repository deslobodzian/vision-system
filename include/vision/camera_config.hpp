//
// Created by ubuntuvm on 11/9/22.
//

#ifndef VISION_SYSTEM_CAMERA_CONFIG_HPP
#define VISION_SYSTEM_CAMERA_CONFIG_HPP

#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

#define USE_MATH_DEFINES_

struct fov {
    double horizontal;
    double vertical;
    double diagonal;
    fov() = default;
    fov(double h, double v) {
        horizontal = h * M_PI / 180.0;
        vertical = v * M_PI / 180.0;
    }
    fov(double h, double v, bool rad) {
        horizontal = h;
        vertical = v;
    }
};

struct IntrinsicParameters {
    double fx;
    double fy;
    double cx;
    double cy;
};

struct resolution {
    unsigned int height;
    unsigned int width;
    resolution() = default;
    resolution(unsigned int w, unsigned int h) {
        height = h;
        width = w;
    }
};

class CameraConfig {

private:
    std::string device_id_;
    fov field_of_view_;
    int frames_per_second_;
    resolution camera_resolution_;
    IntrinsicParameters intrinsic_parameters_;
    std::string pipeline_;

public:
    CameraConfig() = default;

    CameraConfig(
            std::string device,
            double diagonal_fov,
            resolution res,
            int fps,
            IntrinsicParameters intrinsic_parameters
    ) {
        device_id_ = device;
        double d_fov = diagonal_fov * M_PI / 180.0;
        double aspect = hypot(res.width, res.height);
        double h_fov = atan(tan(d_fov / 2.0) * (res.width / aspect)) * 2;
        double v_fov = atan(tan(d_fov / 2.0) * (res.height / aspect)) * 2;
        field_of_view_ = fov(h_fov, v_fov, false);
        camera_resolution_ = res;
        intrinsic_parameters_ = intrinsic_parameters;
        frames_per_second_ = fps;
    }

    CameraConfig(std::string device, fov fov, resolution res, int fps) {
        device_id_ = device;
        field_of_view_ = fov;
        camera_resolution_ = res;
        frames_per_second_ = fps;
    }

    std::string get_device_id() {
        return device_id_;
    }

    fov get_fov() {
        return field_of_view_;
    }

    resolution get_camera_resolution() {
        return camera_resolution_;
    }

    IntrinsicParameters get_intrinsic_parameters() {
        return intrinsic_parameters_;
    };

    int get_fps() {
        return frames_per_second_;
    }
    std::string get_pipeline() {
        pipeline_ = "v4l2src device=" + device_id_ + " ! video/x-raw(memory::NVMM), format=BGR, width=" + std::to_string(camera_resolution_.width) +
                    ", height=" + std::to_string(camera_resolution_.height) + ", framerate=" + std::to_string(frames_per_second_) +
                    "/1 ! appsink";
        return pipeline_;
    }
};
#endif //VISION_SYSTEM_CAMERA_CONFIG_HPP
