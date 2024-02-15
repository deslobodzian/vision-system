#ifndef VISION_SYSTEM_APRILTAG_TRACKER
#define VISION_SYSTEM_APRILTAG_TRACKER

#include <vector>
#include <unordered_map>
#include "apriltag_detector.hpp"
#include "zed.hpp"

class ApriltagTracker {
public:
    ApriltagTracker();

    void update(const std::vector<ZedAprilTag>& current_detections, float timestamp);
    std::unordered_map<int, sl::float3> calculate_velocities() const;

private:
    struct TagHistory {
        std::vector<sl::float4> positions;
        std::vector<float> timestamps;
    };

    std::unordered_map<int, TagHistory> tag_histories;

    void add_to_history(int tag_id, const sl::float4& position, float timestamp);
    sl::float3 calculate_velocity(const TagHistory& history) const;
};


#endif /* VISION_SYSTEM_APRILTAG_TRACKER */
