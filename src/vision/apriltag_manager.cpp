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
	}
}

void AprilTagManager::detector_monocular(MonocularCamera &camera) {
}