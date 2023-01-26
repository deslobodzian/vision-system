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

    image_pub_ = new image_publishable;
    zmq_manager_->create_publisher(image_pub_, "tcp://*:5556");
    vision_pub_ = new vision_publishable;
    zmq_manager_->create_publisher(vision_pub_, "tcp://*:5557");
}

void VisionRunner::run() {
    std::vector<tracked_target_info> vec;
    inference_manager_->inference_on_device(zed_camera_);
    if (image_pub_ != nullptr) {
        cv::Mat img = slMat_to_cvMat(zed_camera_->get_left_image());
//        printf("Mat {w: %i, h %i}\n", img.rows, img.cols);
        cv::Mat img_new;
        cv::cvtColor(img, img_new, cv::COLOR_BGRA2BGR);
        zed_camera_->retrieve_objects(objs_);
        for (auto& object : objs_.object_list) {
            cv::Point p1(object.bounding_box_2d.at(0).x, object.bounding_box_2d.at(0).y);
            cv::Point p2(object.bounding_box_2d.at(2).x, object.bounding_box_2d.at(2).y);
            cv::Rect r(p1, p2);
            std::string id = std::to_string(object.id);
            cv::rectangle(img_new, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img_new, id, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        image_pub_->img_ = img_new;
    }
    if (vision_pub_ != nullptr) {
        for (int i = 0; i < 3; i++) {
            tracked_target_info t(1.0f, 3.0f,2.0f,i);
            vec.push_back(t);
        }
        vision_pub_->targets_ = vec;
    }
    zmq_manager_->send_publishers();
}

VisionRunner::~VisionRunner() {
    delete image_pub_;
    delete vision_pub_;
}
