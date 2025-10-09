#pragma once
#include "apriltag.h"
#include "tag36h11.h"
#include "common/zarray.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct TagDetection {
    int id;
    double center[2];      // Center x, y
    double corners[4][2];  // Four corner points
    double decision_margin;
    double hamming;
};

class TagDetector {
public:
    TagDetector();
    ~TagDetector();
    
    TagDetector(const TagDetector& other);
    TagDetector& operator=(const TagDetector& other);
    
    TagDetector(TagDetector&& other) noexcept;
    TagDetector& operator=(TagDetector&& other) noexcept;
    
    std::vector<TagDetection> detect(const cv::Mat& image);
    zarray_t* detect_raw(const cv::Mat& image); 
    
    void clear_detections();
    
private:
    apriltag_detector_t* detector_ = nullptr;
    apriltag_family_t* tag_family_ = nullptr;
    zarray_t* detections_ = nullptr;
    
    void copy_settings(const apriltag_detector_t* src);
    image_u8_t* mat_to_image_u8(const cv::Mat& mat);
};