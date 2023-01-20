//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"

VisionContainer::VisionContainer() {}

void VisionContainer::init() {
    info("[VisionContainer]: Starting Zmq Manager");
    zmq_manager_ = new ZmqManager();

    info("[VisionContainer]: Setting up zed camera");
    zed_config zed_config{};
    zed_config.res = sl::RESOLUTION::VGA;
    zed_config.fps = 100;
    zed_config.depth_mode = sl::DEPTH_MODE::ULTRA;
    zed_config.sdk_verbose = true;
    zed_config.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    zed_config.units = sl::UNIT::METER;
    zed_config.max_depth = 20.0;
    zed_config.reference_frame = REFERENCE_FRAME::CAMERA;
    zed_config.enable_tracking = true;
    zed_config.enable_mask_output = false;
    zed_config.model = sl::DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

    zed_camera_ = new Zed(zed_config);

    info("[Vision Container]: Starting Inference Manager");
    inference_manager_ = new InferenceManager("../engines/best.engine");
    inference_manager_->init();

//    info("[VisionContainer]: Setting up AprilTag manager");
//    detector_config apriltag_config {};
//    apriltag_config.tf = tag16h5;
//    apriltag_config.quad_decimate = 1;
//    apriltag_config.quad_sigma = 0.5;
//    apriltag_config.nthreads = 2;
//    apriltag_config.debug = false;
//    apriltag_config.refine_edges = true;

//    tag_manager_ = new AprilTagManager<float>(apriltag_config);


}

void VisionContainer::detect_zed_targets() {
//    tag_manager_->detect_tags_zed(zed_camera_);
    inference_manager_->inference_on_device(zed_camera_);
}


void VisionContainer::run() {
    init();
    info("[VisionContainer]: Starting system");

    vision_runner_ = new VisionRunner(&task_manager_, 0.05, "vision-runner");

    vision_runner_->zed_camera_ = zed_camera_;
    vision_runner_->inference_manager_ = inference_manager_;
    vision_runner_->zmq_manager_ = zmq_manager_;

    vision_runner_->init();
    vision_runner_->start();

    for (;;) {
        usleep(1000000);
    }
}

VisionContainer::~VisionContainer() {
    delete vision_runner_;
    delete zed_camera_;
    delete inference_manager_;
    delete zmq_manager_;
}
