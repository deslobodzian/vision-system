//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"

VisionRunner::VisionRunner(
        TaskManager* manager,
        double period,
        const std::string &name):
        Task(manager, period, name) {
}

void VisionRunner::init() {
    info("initializing [VisionRunner]");
    zed_camera_->open_camera();

    image_pub_ = new image_publishable;
    zmq_manager_->create_publisher(image_pub_, "tcp://*:5556");
}

void VisionRunner::run() {
    inference_manager_->inference_on_device(zed_camera_);
    if (image_pub_ != nullptr) {
        image_pub_->img_ = slMat_to_cvMat(zed_camera_->get_left_image());
    }
    zmq_manager_->send_publishers();
}

VisionRunner::~VisionRunner() {
    delete image_pub_;
}
