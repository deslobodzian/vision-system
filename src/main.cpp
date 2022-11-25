//
// Created by DSlobodzian on 1/2/2022.
//
#include "utils/utils.hpp"
#include "vision/monocular_camera.hpp"
#include "vision/Zed.hpp"
#include "vision/camera_config.hpp"
#include "vision/apriltag_manager.hpp"

int main() {
    // Instantiate Systems
    AprilTagManager at_manager;
    std::vector<TrackedTargetInfo> zed_targets_;

    // generate camera configs
    IntrinsicParameters<float> parameters{1116.821, 1113.573, 678.58, 367.73};
    resolution res(320, 240);
    CameraConfig config(
            "/dev/video0", // need to find a better way for device id.
            68.5,
            res,
            30,
            parameters
            );
//
//    // monocular cameras
    MonocularCamera life_cam_(config);
    life_cam_.open_camera();

    // zed camera;
    Zed zed_;

    // opening zed camera. Do this before zed inference thread.
    zed_.open_camera();
    zed_.enable_tracking();

    at_manager.add_detector_thread(zed_);
    at_manager.add_detector_thread(life_cam_);

    std::string prev_string;

    while (true) {
        zed_targets_ = at_manager.get_zed_targets();
        if (!zed_targets_.empty()) {
            TrackedTargetInfo tmp = zed_targets_.at(0);
	    std::string s = "Target {x: " + std::to_string(tmp.get_x()) +
            ", y: " + std::to_string(tmp.get_y()) +
            ", z: " + std::to_string(tmp.get_z()) +
            "} angle: " + std::to_string(tmp.get_yaw_angle());
	    if (s != prev_string) {
            info("Targets found: " + std::to_string(at_manager.get_monocular_targets().size() + at_manager.get_zed_targets().size()));
//            at_manager.print_monocular_dt();
//            at_manager.print_zed_dt();
//		    info(s);
	    }
	    prev_string = s;
        }
    }
}

