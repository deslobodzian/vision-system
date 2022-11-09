//
// Created by DSlobodzian on 1/2/2022.
//
#include "utils.hpp"
#include "networking/udp_server.hpp"
#include "vision/monocular_camera.hpp"
#include "vision/Zed.hpp"
#include "vision/camera_config.hpp"


int main() {

    // Instantiate Systems
//    UDPServer server;

    // generate camera configs
    IntrinsicParameters parameters{1116.821, 1113.573, 678.58, 367.73};
    resolution res(1280, 720);
    CameraConfig config(
            "usb-Microsoft_Microsoft®_LifeCam_HD-3000-video-index0",
            68.5,
            res,
            30,
            parameters
            );

    // monocular cameras
    MonocularCamera life_cam_(config);
    life_cam_.open_camera();

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
        life_cam_.read_frame();
//        cv::Mat img = slMat_to_cvMat(zed_.get_left_image());
        cv::Mat img = life_cam_.get_frame();
        detector.fetch_detections(img);

        if (detector.has_targets()) {
            apriltag_detection_t* det = detector.get_target_from_id(1);
            if (det != nullptr) {
                cv::Point center = detector.get_detection_center(det);
//                sl::float3 pos = zed_.get_position_from_pixel(center.x, center.y);
                info ("Target with ID 1 has center x: " +
                    std::to_string(center.x)
                    + ", y: " +
                    std::to_string(center.y)
                );
//                info ("Coordinates of {x: " +
//                std::to_string(pos.x) +
//                ", y: " +
//                std::to_string(pos.y) +
//                ", z:" +
//                std::to_string(pos.z) +
//                "}"
//                );
//                info ("Target distance: " + std::to_string(zed_.get_distance_from_point(pos)));
            }
        }
//        info("Targets: id 1" + std::to_string(detector.get_current_number_of_targets()));
        //cv::imshow("Zed", img);
	//if(cv::waitKey(30) >= 0) break;
    }
}

