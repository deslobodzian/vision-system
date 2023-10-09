#ifndef VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP
#define VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP

#include <string>
#include "inference/yolo.hpp"
#include "inference/inference_utils.hpp"
#include <sl/Camera.hpp>
#include <cmath>

#define NMS_THRESH 0.4
#define CONF_THRESH 0.3

class DetectionsPlayback {
public:
    DetectionsPlayback(std::string svo_file);
    ~DetectionsPlayback();
    
    void detect();
    void export_video();
    
private:
    Yolo yolo_;
    sl::Camera zed_;
    sl::Mat left_sl;
    cv::Mat left_cv;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Resolution display_resolution;
    cv::VideoWriter video_writer;
};

#endif /* VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP */