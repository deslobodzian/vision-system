#pragma once

#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

class CVCamera {
public:
    CVCamera(const int& id);
    bool fetch();
    cv::Mat get_image() const { return image_mat_; }
private:
    int id_;
    cv::VideoCapture camera_;
    cv::Mat image_mat_{};
};
