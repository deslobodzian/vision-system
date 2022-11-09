//
// Created by DSlobodzian on 1/2/2022.
//
#include "utils.hpp"
#include "networking/udp_server.hpp"
#include "vision/Zed.hpp"


int main() {

    // Instantiate Systems
//    UDPServer server;

    // generate camera configs
    // resolution res(1920, 1080);
    // CameraConfig config("/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_EEDAD4AF-video-index0", 72, res, 30);

    // monocular cameras
    // MonocularCamera c920_(config);

    // zed camera;
    Zed zed_;

    // opening zed camera. Do this before zed inference thread.
    zed_.open_camera();
    zed_.enable_tracking();


//    zed_.enable_object_detection();

    DetectorConfig cfg = {
            tag36h11,
            0.5,
            0.5,
            2,
            false,
            true
    };
    TagDetector detector(cfg);
    while (true) {
        zed_.fetch_measurements();
        cv::Mat img = slMat_to_cvMat(zed_.get_left_image());
        detector.fetch_detections(img);

        if (detector.has_targets()) {
            apriltag_detection_t* det = detector.get_target_from_id(1);
            if (det != nullptr) {
                cv::Point center = detector.get_detection_center(det);
                info ("Target with ID 1 has center x: " +
                    std::to_string(center.x)
                    + ", y: " +
                    std::to_string(center.y)
                );
            }
        }
//        info("Targets: id 1" + std::to_string(detector.get_current_number_of_targets()));
        //cv::imshow("Zed", img);
	//if(cv::waitKey(30) >= 0) break;
    }
}

