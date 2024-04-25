#ifdef WITH_CUDA
#include "vision/apriltag_tracker.hpp"

void ApriltagTracker::update(const std::vector<ZedAprilTag>& current_detections,
                             float timestamp) {
  for (const auto& detection : current_detections) {
    sl::float4 center_position = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
      center_position.x += detection.corners[i].x;
      center_position.y += detection.corners[i].y;
      center_position.z += detection.corners[i].z;
    }
    center_position.x /= 4.0f;
    center_position.y /= 4.0f;
    center_position.z /= 4.0f;

    add_to_history(detection.tag_id, center_position, timestamp);
  }
}

std::unordered_map<int, sl::float3> ApriltagTracker::calculate_velocities()
    const {
  std::unordered_map<int, sl::float3> velocities;
  for (const auto& [tag_id, history] : tag_histories) {
    velocities[tag_id] = calculate_velocity(history);
  }
  return velocities;
}

void ApriltagTracker::add_to_history(int tag_id, const sl::float4& position,
                                     float timestamp) {
  auto& history = tag_histories[tag_id];
  history.positions.push_back(position);
  history.timestamps.push_back(timestamp);

  if (history.positions.size() > 10) {  // Keep last 10 samples
    history.positions.erase(history.positions.begin());
    history.timestamps.erase(history.timestamps.begin());
  }
}

sl::float3 ApriltagTracker::calculate_velocity(
    const TagHistory& history) const {
  if (history.positions.size() < 2) {
    return sl::float3{0, 0, 0};
  }

  auto& pos_start = history.positions.front();
  auto& pos_end = history.positions.back();
  float time_start = history.timestamps.front();
  float time_end = history.timestamps.back();
  float time_diff = time_end - time_start;

  if (time_diff == 0) {
    return sl::float3{0, 0, 0};
  }

  sl::float3 velocity;
  velocity.x = (pos_end.x - pos_start.x) / time_diff;
  velocity.y = (pos_end.y - pos_start.y) / time_diff;
  velocity.z = (pos_end.z - pos_start.z) / time_diff;

  return velocity;
}
#endif /* WITH_CUDA */
