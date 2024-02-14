#ifdef WITH_CUDA
#ifndef VISION_SYSTEM_APRILTAG_DETECTOR_HPP
#define VISION_SYSTEM_APRILTAG_DETECTOR_HPP

#include "cuAprilTags.h"
#include <cuda_runtime.h> 
#include <opencv2/opencv.hpp>
#include <vector>

class ApriltagDetector {
public:
    ApriltagDetector(uint32_t img_width, uint32_t img_height, uint32_t tile_size, cuAprilTagsFamily tag_family, float tag_dim);
    ~ApriltagDetector();

    ApriltagDetector(const ApriltagDetector&) = delete;
    ApriltagDetector& operator=(const ApriltagDetector&) = delete;

    std::vector<cuAprilTagsID_t> detectTags(const cuAprilTagsImageInput_t& img_input);
    std::vector<cuAprilTagsID_t> detectAprilTagsInCvImage( const cv::Mat& cvImage); 

private:
    cuAprilTagsHandle hApriltags = nullptr;
    uint32_t maxTags = 1000;
};

#endif /* VISION_SYSTEM_APRILTAG_DETECTOR_HPP */
#endif /* WITH_CUDA */
