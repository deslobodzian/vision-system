#include "vision/apriltag_manager.hpp"


void AprilTagManager::add_detector_thread(Zed & camera) {
	threads_.emplace_back(&AprilTagManager::detector_zed, this, std::ref(camera));
}

void AprilTagManager::add_detector_thread(MonocularCamera & camera) {
	threads_.emplace_back(&AprilTagManager::detector_monocular, this, std::ref(camera));
}

void AprilTagManager::detector_zed(Zed & camera) {
	apriltag_family_t* tf = tag36h11_create();
	apriltag_detector_t* td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);

	cv::Mat frame, gray;
	while (true) {
//		camera.get_left_image(&frame);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		image_u8_t im = {
			.width = gray.cols,
			.height = gray.rows,
			.stride = gray.cols,
			.buf = gray.data
		};

		zarray_t* detections = apriltag_detector_detect(td, &im);
	}
}

void AprilTagManager::detector_monocular(MonocularCamera &camera) {
    apriltag_family_t* tf = tag36h11_create();
	apriltag_detector_t* td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);

	cv::Mat frame, gray;
//	while (true) {
//		camera.get_left_image(&frame);
//		cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
//
//		image_u8_t im = {
//			.width = gray.cols,
//			.height = gray.rows,
//			.stride = gray.cols,
//			.buf = gray.data
//		}
//
//		zarray_t* detections = apriltag_detector_detect(td, &im);
//	}
}