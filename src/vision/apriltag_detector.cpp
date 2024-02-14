#include "vision/apriltag_detector.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include "utils/timer.h"


bool convertMatToCuAprilTagsImageInput(const cv::Mat& image, cuAprilTagsImageInput_t& img_input) {
    if (image.type() != CV_8UC3) {
        std::cerr << "Image format is not compatible. Expected 8-bit, 3-channel BGR." << std::endl;
        return false;
    }

    uchar3* dev_image = nullptr;
    size_t pitch = 0;
    cudaError_t cudaStatus = cudaMallocPitch(&dev_image, &pitch, image.cols * sizeof(uchar3), image.rows);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMallocPitch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    cudaStatus = cudaMemcpy2D(dev_image, pitch, image.ptr<uchar>(), image.step, image.cols * sizeof(uchar3), image.rows, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy2D failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(dev_image); 
        return false;
    }

    img_input.dev_ptr = dev_image;
    img_input.pitch = pitch;
    img_input.width = static_cast<uint16_t>(image.cols);
    img_input.height = static_cast<uint16_t>(image.rows);

    return true;
}

ApriltagDetector::ApriltagDetector(uint32_t img_width, uint32_t img_height, uint32_t tile_size, cuAprilTagsFamily tag_family, float tag_dim) {
    if (nvCreateAprilTagsDetector(&hApriltags, img_width, img_height, tile_size, tag_family, nullptr, tag_dim) != 0) {
        throw std::runtime_error("Failed to create AprilTags detector");
    }
}

ApriltagDetector::~ApriltagDetector() {
    if (hApriltags && cuAprilTagsDestroy(hApriltags) != 0) {
        std::cerr << "Failed to destroy AprilTags detector" << std::endl;
    }
}

std::vector<cuAprilTagsID_t> ApriltagDetector::detectTags(const cuAprilTagsImageInput_t& img_input) {
    std::vector<cuAprilTagsID_t> detectedTags;
    detectedTags.resize(maxTags); 

    uint32_t num_tags_detected;
    if (cuAprilTagsDetect(hApriltags, &img_input, detectedTags.data(), &num_tags_detected, maxTags, nullptr) != 0) {
        throw std::runtime_error("Failed to detect AprilTags");
    }
    LOG_DEBUG("Number Detected Tags: ", num_tags_detected);

    detectedTags.resize(num_tags_detected); 
    return detectedTags;
}

std::vector<cuAprilTagsID_t> ApriltagDetector::detectAprilTagsInCvImage(const cv::Mat& cvImage) {
    cuAprilTagsImageInput_t img_input;
    if (!convertMatToCuAprilTagsImageInput(cvImage, img_input)) {
        throw std::runtime_error("Failed to convert OpenCV image to cuAprilTagsImageInput_t");
    }

    Timer t;
    t.start();

    std::vector<cuAprilTagsID_t> detectedTags = detectTags(img_input);

    LOG_DEBUG("Timer took: ", t.get_nanoseconds(), "ns");

    cudaFree(img_input.dev_ptr);

    return detectedTags;
}

