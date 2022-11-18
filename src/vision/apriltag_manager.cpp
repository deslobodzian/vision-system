#include "vision/apriltag_manager.hpp"

void AprilTagManager::add_detector_thread(Zed &camera) {
	threads_.emplace_back(&AprilTagManager::detector_zed, this, std::ref(camera));
}

void AprilTagManager::add_detector_thread(MonocularCamera &camera) {
	threads_.emplace_back(&AprilTagManager::detector_monocular, this, std::ref(camera));
}

void AprilTagManager::detector_zed(Zed &camera) {
    DetectorConfig cfg = {
            tag36h11,
            0.0,
            0.0,
            1,
            false,
            true
    };
    zed_detector_ = TagDetector(cfg);

	while (true) {
        camera.fetch_measurements();
        zed_detector_.fetch_detections(slMat_to_cvMat(camera.get_left_image()));
        std::vector<TrackedTargetInfo> targets;

        apriltag_detector_t *det;
        for (int i = 0; i < zed_detector_.get_current_number_of_targets(); i++) {
            zarray_get(zed_detector_.get_current_detections(), i, &det);
            Corners c = zed_detector_.get_detection_corners(reinterpret_cast<apriltag_detection_t *>(det));
            sl::float3 tr = camera.get_position_from_pixel(c.tr);
            sl::float3 tl = camera.get_position_from_pixel(c.tl);
            sl::float3 br = camera.get_position_from_pixel(c.br);
            sl::Pose pose = zed_detector_.get_estimated_target_pose(tr, tl, br);

        }
        sl::float3 tl = camera.get_position_from_pixel()
	}
}

void AprilTagManager::detector_monocular(MonocularCamera &camera) {
}