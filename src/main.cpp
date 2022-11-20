//
// Created by DSlobodzian on 1/2/2022.
//
#include "utils.hpp"
#include "networking/udp_server.hpp"
#include "vision/monocular_camera.hpp"
#include "vision/Zed.hpp"
#include "vision/camera_config.hpp"
#include "vision/apriltag_manager.hpp"


int main() {
    // Instantiate Systems
//    UDPServer server;
    AprilTagManager at_manager;
    std::vector<TrackedTargetInfo> zed_targets_;

    // generate camera configs
//    IntrinsicParameters parameters{1116.821, 1113.573, 678.58, 367.73};
//    resolution res(1280, 720);
//    CameraConfig config(
//            "/dev/video1", // need to find a better way for device id.
////            "usb-Microsoft_Microsoft®_LifeCam_HD-3000-video-index0",
//            68.5,
//            res,
//            30,
//            parameters
//            );
//
//    // monocular cameras
//    MonocularCamera life_cam_(config);
//    life_cam_.open_camera();

    // zed camera;
    Zed zed_;

    // opening zed camera. Do this before zed inference thread.
    zed_.open_camera();
    zed_.enable_tracking();

    at_manager.add_detector_thread(zed_);

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        zed_targets_ = at_manager.get_zed_targets();
        if (!zed_targets_.empty()) {
            TrackedTargetInfo tmp = zed_targets_.at(0);
//            info("Target {x: " + std::to_string(tmp.get_x()) +
//            ", y: " + std::to_string(tmp.get_y()) +
//            ", z: " + std::to_string(tmp.get_z()) +
//            "} angle: " + std::to_string(tmp.get_yaw_angle()));
        }
//        info("Zed has found " + std::to_string(zed_targets_.size()) + " targets!");
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//        info("Zed thread took " + std::to_string(duration.count()) + " milliseconds");
    }
//    zed_.enable_object_detection();

//    DetectorConfig cfg = {
//            tag36h11,
//            0.5,
//            0.5,
//            2,
//            false,
//            true
//    };
//    TagDetector detector_life(cfg);
//    TagDetector detector_zed(cfg);
//    while (true) {
//        zed_.fetch_measurements();
//        life_cam_.read_frame();
//        cv::Mat img = life_cam_.get_frame();
//        detector_life.fetch_detections(img);
//
//        if (detector_life.has_targets()) {
//            apriltag_detection_t* det = detector_life.get_target_from_id(1);
//            if (det != nullptr) {
//                cv::Point center = detector_life.get_detection_center(det);
//                apriltag_pose_t pose = detector_life.get_estimated_target_pose(life_cam_.get_intrinsic_parameters(), det, 0.1651);
////                sl::float3 pos = zed_.get_position_from_pixel(center.x, center.y);
////                info ("Target with ID 1 has center x: " +
////                    std::to_string(center.x)
////                    + ", y: " +
////                    std::to_string(center.y)
////                );
//                info ("Coordinates of {x: " +
//                std::to_string(pose.t->data[0]) +
//                ", y: " +
//                std::to_string(pose.t->data[1]) +
//                ", z:" +
//                std::to_string(pose.t->data[2]) +
//                "}"
//                );
////                info ("Target distance: " + std::to_string(zed_.get_distance_from_point(pos)));
//            }
//        }
//        img = slMat_to_cvMat(zed_.get_left_image());
//        detector_zed.fetch_detections(img);
//        if (detector_zed.has_targets()) {
//            apriltag_detection_t* det = detector_zed.get_target_from_id(1);
//            if (det != nullptr) {
//                cv::Point center = detector_zed.get_detection_center(det);
//                sl::float3 pos = zed_.get_position_from_pixel(center.x, center.y);
////                info ("Target with ID 1 has center x: " +
////                    std::to_string(center.x)
////                    + ", y: " +
////                    std::to_string(center.y)
////                );
//                info ("Coordinates of {x: " +
//                std::to_string(pos.x) +
//                ", y: " +
//                std::to_string(pos.y) +
//                ", z:" +
//                std::to_string(pos.z) +
//                "}"
//                );
////                info ("Target distance: " + std::to_string(zed_.get_distance_from_point(pos)));
//            }
//        }
////        info("Targets: id 1" + std::to_string(detector.get_current_number_of_targets()));
//        //cv::imshow("Zed", img);
//	//if(cv::waitKey(30) >= 0) break;
//    }
}

