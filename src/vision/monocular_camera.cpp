#include "vision/monocular_camera.hpp"

MonocularCamera::MonocularCamera() {}

int MonocularCamera::open_camera() {
  cap_.open(0, cv::CAP_ANY);
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  cap_.set(cv::CAP_PROP_FPS, 30);
  return cap_.isOpened();
}

void MonocularCamera::close() { cap_.release(); }

void MonocularCamera::fetch_measurements() { cap_.read(frame_); }

const cv::Mat MonocularCamera::get_frame() const { return frame_; }
const CAMERA_TYPE MonocularCamera::get_camera_type() const { return MONOCULAR; }

MonocularCamera::~MonocularCamera() { close(); }
