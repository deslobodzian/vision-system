#include "vision/apriltag_detector.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include "utils/timer.h"
#include "preprocess_kernels.h"

bool convert_mat_to_cu_april_tags_image_input(const cv::Mat& image, cuAprilTagsImageInput_t& img_input) {
    if (image.type() != CV_8UC3) {
        std::cerr << "Image format is not compatible. Expected 8-bit, 3-channel BGR." << std::endl;
        return false;
    }

    uchar3* dev_image = nullptr;
    size_t pitch = 0;
    cudaError_t cuda_status = cudaMallocPitch(&dev_image, &pitch, image.cols * sizeof(uchar3), image.rows);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMallocPitch failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    cuda_status = cudaMemcpy2D(dev_image, pitch, image.ptr<uchar>(), image.step, image.cols * sizeof(uchar3), image.rows, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMemcpy2D failed: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(dev_image);
        return false;
    }

    img_input.dev_ptr = dev_image;
    img_input.pitch = pitch;
    img_input.width = static_cast<uint16_t>(image.cols);
    img_input.height = static_cast<uint16_t>(image.rows);

    return true;
}

ApriltagDetector::ApriltagDetector() {}

void ApriltagDetector::init_detector(uint32_t img_width, uint32_t img_height, uint32_t tile_size, cuAprilTagsFamily tag_family, float tag_dim) {
    if (nvCreateAprilTagsDetector(&h_apriltags, img_width, img_height, tile_size, tag_family, nullptr, tag_dim) != 0) {
        throw std::runtime_error("Failed to create AprilTags detector");
    }
}

ApriltagDetector::~ApriltagDetector() {
    if (h_apriltags && cuAprilTagsDestroy(h_apriltags) != 0) {
        std::cerr << "Failed to destroy AprilTags detector" << std::endl;
    }
    cudaFree(input_image_.dev_ptr);
}

std::vector<cuAprilTagsID_t> ApriltagDetector::detect_tags(const cuAprilTagsImageInput_t& img_input) {
    std::vector<cuAprilTagsID_t> detected_tags;
    detected_tags.resize(max_tags);

    uint32_t num_tags_detected;
    if (cuAprilTagsDetect(h_apriltags, &img_input, detected_tags.data(), &num_tags_detected, max_tags, nullptr) != 0) {
        throw std::runtime_error("Failed to detect AprilTags");
    }
    LOG_DEBUG("Number Detected Tags: ", num_tags_detected);

    detected_tags.resize(num_tags_detected);
    return detected_tags;
}

std::vector<cuAprilTagsID_t> ApriltagDetector::detect_april_tags_in_cv_image(const cv::Mat& cv_image) {
    if (!convert_mat_to_cu_april_tags_image_input(cv_image, input_image_)) {
        throw std::runtime_error("Failed to convert OpenCV image to cu_april_tags_image_input_t");
    }

    Timer t;
    t.start();

    std::vector<cuAprilTagsID_t> detected_tags = detect_tags(input_image_);

    LOG_DEBUG("Timer took: ", t.get_nanoseconds(), "ns");


    return detected_tags;
}

std::vector<cuAprilTagsID_t> ApriltagDetector::detect_april_tags_in_sl_image(const sl::Mat& sl_image, CUstream_st* stream) {
    Timer t;
    convert_sl_mat_to_april_tag_input(sl_image, input_image_, stream);
    LOG_DEBUG("Mat conversion took: ", t.get_nanoseconds(), " ns");

    t.start();
    std::vector<cuAprilTagsID_t> detected_tags = detect_tags(input_image_);
    LOG_DEBUG("Tag detection took: ", t.get_nanoseconds(), " ns");

    return detected_tags;
}

std::vector<ZedAprilTag> ApriltagDetector::calculate_zed_apriltag(const sl::Mat& point_cloud, const std::vector<cuAprilTagsID_t>& detections) {
    std::vector<ZedAprilTag> zed_tags;

    for (const auto& tag : detections) {
        ZedAprilTag z_tag;
        for (int i = 0; i < 4; ++i) {
            sl::float4 point3D;
            LOG_INFO("Corner ", i, ",: {", tag.corners[i].x, ", ", tag.corners[i].y, "}");
            point_cloud.getValue(tag.corners[i].x, tag.corners[i].y, &point3D);
            z_tag.corners[i] = point3D;
        }
        z_tag.tag_id = tag.id;
    }

    return zed_tags;
}
