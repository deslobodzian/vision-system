#ifdef WITH_CUDA
#ifndef VISION_SYSTEM_APRILTAG_DETECTOR_HPP
#define VISION_SYSTEM_APRILTAG_DETECTOR_HPP

#include "cuAprilTags.h"
#include <cuda_runtime.h> 
#include <opencv2/opencv.hpp>
#include <vector>
#include "vision/zed.hpp"
#include "utils/timer.h"


typedef struct {
    /* xyz corners */
    sl::float4 corners[4];
    sl::float4 center;
    sl::Orientation orientation;
    int tag_id;
} ZedAprilTag;

class ApriltagDetector {
public:
    ApriltagDetector();
    void init_detector(uint32_t img_width, uint32_t img_height, uint32_t tile_size, cuAprilTagsFamily tag_family, float tag_dim);
    ~ApriltagDetector();

    ApriltagDetector(const ApriltagDetector&) = delete;
    ApriltagDetector& operator=(const ApriltagDetector&) = delete;

    std::vector<cuAprilTagsID_t> detect_tags(const cuAprilTagsImageInput_t& img_input);
    std::vector<cuAprilTagsID_t> detect_april_tags_in_cv_image( const cv::Mat& cvImage); 
    std::vector<cuAprilTagsID_t> detect_april_tags_in_sl_image(const sl::Mat& sl_image);
    std::vector<ZedAprilTag> calculate_zed_apriltag(const sl::Mat& point_cloud, const sl::Mat& normals, const std::vector<cuAprilTagsID_t>& detetions);

private:
    Timer timer_;
    cuAprilTagsHandle h_apriltags = nullptr;
    cuAprilTagsImageInput_t input_image_;
    cudaStream_t cuda_stream_;
    const uint32_t max_tags;
};

#endif /* VISION_SYSTEM_APRILTAG_DETECTOR_HPP */
#endif /* WITH_CUDA */
