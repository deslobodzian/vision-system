#ifndef VISION_SYSTEM_MONOCULAR_CAMERA_HPP
#define VISION_SYSTEM_MONOCULAR_CAMERA_HPP

#include "i_camera.hpp"
#include <opencv2/opencv.hpp>

class MonocularCamera : public ICamera {
public:
    MonocularCamera();
    ~MonocularCamera();
    int open_camera() override;
    void fetch_measurements() override;
    const cv::Mat get_frame() const;
    const CAMERA_TYPE get_camera_type() const override;
    
private:
    cv::VideoCapture cap_;
    cv::Mat frame_;
};

#endif /* VISION_SYSTEM_MONOCULAR_CAMERA_HPP */