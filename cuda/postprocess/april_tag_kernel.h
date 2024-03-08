#ifndef APRIL_TAG_KERNEL_H
#define APRIL_TAG_KERNEL_H

#include "cuAprilTags.h"
#include "sl/Camera.hpp"
#include "vector"

struct DeviceOrientation {
    float data[4];
};

struct ZedAprilTag {
    sl::float4 corners[4];
    sl::float4 center;
    DeviceOrientation orientation;
    uint16_t tag_id;
};

std::vector<ZedAprilTag>
detect_and_calculate(const sl::Mat &point_cloud, const sl::Mat &normals,
        const std::vector<cuAprilTagsID_t> &detections,
        cuAprilTagsID_t *gpu_detections, ZedAprilTag *gpu_zed_tags,
        int max_detections);

#endif // APRIL_TAG_KERNEL_H
