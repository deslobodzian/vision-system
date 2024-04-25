#ifdef WITH_CUDA
#ifndef VISION_SYSTEM_APRIL_TAG_UTILS_HPP
#define VISION_SYSTEM_APRIL_TAG_UTILS_HPP

#include "logger.hpp"
#include "vision/apriltag_detector.hpp"
#include "vision/zed.hpp"

namespace AprilTagUtils {

static std::vector<ZedAprilTag>
calculate_zed_apriltags(ZedCamera &camera, ApriltagDetector &tag_detector) {

  auto tags =
      tag_detector.detect_april_tags_in_sl_image(camera.get_left_image());
  auto point_cloud = camera.get_point_cloud();

  std::vector<ZedAprilTag> zed_tags;

  for (const auto &tag : tags) {
    ZedAprilTag z_tag;
    sl::float3 average_normal = {0, 0, 0};
    sl::uint2 center_px = {0, 0};

    for (int i = 0; i < 4; ++i) {
      sl::float4 point3D;
      point_cloud.getValue(tag.corners[i].x, tag.corners[i].y, &point3D);
      z_tag.corners[i] = point3D;
      z_tag.center += point3D;
      center_px.x += tag.corners[i].x;
      center_px.y += tag.corners[i].y;

      sl::float4 corner_normal;
      average_normal +=
          sl::float3(corner_normal.x, corner_normal.y, corner_normal.z);
    }

    z_tag.center /= 4.0f;
    average_normal /= 4.0f;
    center_px /= 4;
    z_tag.tag_id = tag.id;

    sl::Plane plane;
    sl::ERROR_CODE err = camera.get_plane(center_px, plane);
    if (err == sl::ERROR_CODE::SUCCESS) {
      z_tag.plane = plane;
      z_tag.orientation = plane.getPose().getOrientation();
    } else {
      LOG_ERROR("Failed to find plane for AprilTag ", tag.id);
    }

    zed_tags.push_back(z_tag);
  }

  return zed_tags;
}

} // namespace AprilTagUtils

#endif /* VISION_SYSTEM_APRIL_TAG_UTILS_HPP */
#endif /* WITH_CUDA */
