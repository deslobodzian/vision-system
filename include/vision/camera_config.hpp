//
// Created by ubuntuvm on 11/9/22.
//

#ifndef VISION_SYSTEM_CAMERA_CONFIG_HPP
#define VISION_SYSTEM_CAMERA_CONFIG_HPP

#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"

#define USE_MATH_DEFINES_

template <typename T>
struct fov {
    T horizontal;
    T vertical;
    T diagonal;
    fov() = default;
    fov(T h, T v) {
        horizontal = h * M_PI / 180.0;
        vertical = v * M_PI / 180.0;
    }
    fov(T h, T v, bool rad) {
        horizontal = h;
        vertical = v;
    }
};

template <typename T>
struct IntrinsicParameters {
    T fx;
    T fy;
    T cx;
    T cy;
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

template <typename T>
class CameraConfig {
private:
    std::string device_id_;
    fov<T> field_of_view_;
    int frames_per_second_;
    resolution camera_resolution_;
    IntrinsicParameters<T> intrinsic_parameters_;
    std::string pipeline_;

public:
    CameraConfig() = default;

    CameraConfig(
            const std::string &device,
            T diagonal_fov,
            resolution res,
            int fps,
            const IntrinsicParameters<T> &intrinsic_parameters
    ) {
        device_id_ = device;
        T d_fov = diagonal_fov * M_PI / 180.0;
        T aspect = hypot(res.width, res.height);
        T h_fov = atan(tan(d_fov / 2.0) * (res.width / aspect)) * 2;
        T v_fov = atan(tan(d_fov / 2.0) * (res.height / aspect)) * 2;
        field_of_view_ = fov(h_fov, v_fov, false);
        camera_resolution_ = res;
        intrinsic_parameters_ = intrinsic_parameters;
        frames_per_second_ = fps;
    }

    CameraConfig(const std::string &device, fov<T> fov, resolution res, int fps) {
        device_id_ = device;
        field_of_view_ = fov;
        camera_resolution_ = res;
        frames_per_second_ = fps;
    }

    std::string get_device_id() {
        return device_id_;
    }

    fov<T> get_fov() {
        return field_of_view_;
    }

    resolution get_camera_resolution() {
        return camera_resolution_;
    }

    IntrinsicParameters<T> get_intrinsic_parameters() {
        return intrinsic_parameters_;
    };

    int get_fps() const {
        return frames_per_second_;
    }

    std::string get_pipeline() {
        pipeline_ = "v4l2src device=" + device_id_ + " ! video/x-raw(memory::NVMM), format=BGR, width=" + std::to_string(camera_resolution_.width) +
                    ", height=" + std::to_string(camera_resolution_.height) + ", framerate=" + std::to_string(frames_per_second_) +
                    "/1 ! appsink";
        return pipeline_;
    }
};

struct zed_config {
    // Initial Parameters;
    sl::RESOLUTION res;
    int fps;
    sl::DEPTH_MODE depth_mode;
    bool sdk_verbose;
    sl::COORDINATE_SYSTEM coordinate_system;
    sl::UNIT units;
    float max_depth;

    sl::REFERENCE_FRAME reference_frame;

    bool enable_tracking;
    bool enable_mask_output;
    sl::DETECTION_MODEL model;

    float detection_confidence_threshold;
};

#endif //VISION_SYSTEM_CAMERA_CONFIG_HPP
