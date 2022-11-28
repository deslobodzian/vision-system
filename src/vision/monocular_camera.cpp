//
// Created by DSlobodzian on 1/27/2022.
//
#include "vision/monocular_camera.hpp"

template <typename T>
MonocularCamera<T>::MonocularCamera(const CameraConfig<T> &config) : config_(config){}

template <typename T>
MonocularCamera<T>::~MonocularCamera() {
    cap_.release();
}

template <typename T>
int MonocularCamera<T>::open_camera() {
    //std::string c = 
//	    "v4l2src device=/dev/video" + std::to_string(device_id_) +
//	    " ! video/x-raw, width=" + std::to_string(config_.camera_resolution.width) +
//	    ", height=" + std::to_string(config_.camera_resolution.height) +
//	    ", framerate="+ std::to_string(config_.frames_per_second) + 
//	    "/1 ! videoconvert ! appsink";
    cap_.open(config_.get_device_id(), cv::CAP_V4L2);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.get_camera_resolution().width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.get_camera_resolution().height);
    cap_.set(cv::CAP_PROP_FPS, config_.get_fps());
    return cap_.isOpened();
}

template <typename T>
IntrinsicParameters<T> MonocularCamera<T>::get_intrinsic_parameters() {
    return config_.get_intrinsic_parameters();
}

template <typename T>
void MonocularCamera<T>::fetch_measurements() {
    cap_.read(frame_);
}

template <typename T>
cv::Mat MonocularCamera<T>::get_frame() {
    return frame_;
}

template <typename T>
void MonocularCamera<T>::get_frame(cv::Mat& image) {
    cap_.read(image);
}

template <typename T>
void MonocularCamera<T>::draw_rect(const cv::Rect &rect) {
    rectangle(frame_, rect, cv::Scalar(0, 255, 255), 1);
}

template <typename T>
void MonocularCamera<T>::draw_crosshair(const cv::Rect &rect) {
    cv::Point object_center = (rect.br() + rect.tl()) / 2.0;
    cv::Point top = object_center + cv::Point(0, 10);
    cv::Point bot = object_center - cv::Point(0, 10);
    cv::Point left = object_center - cv::Point(10, 0);
    cv::Point right = object_center + cv::Point(10, 0);

    line(frame_, top, bot, cv::Scalar(0, 255, 0), 1);
    line(frame_, left, right, cv::Scalar(0, 255, 0), 1);
}

template <typename T>
int MonocularCamera<T>::get_id() const {
	return device_id_;
}

template class MonocularCamera<float>;
template class MonocularCamera<double>;
