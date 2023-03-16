//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"
#include <chrono>

VisionRunner::VisionRunner(
        TaskManager* manager,
        double period,
        const std::string &name):
        Task(manager, period, name) {
}

void VisionRunner::init() {
    info("Initializing [VisionRunner]");
    std::string ip_address = "10.56.87.20";
//    std::string ip_address = "*";

//    image_pub_ = new image_publishable;
//    zmq_manager_->create_publisher(image_pub_, "tcp://" + ip_address + ":5556");
    vision_pub_ = new vision_publishable;
    zmq_manager_->create_publisher(vision_pub_, "tcp://" + ip_address + ":5557");
}

void VisionRunner::run() {
    auto start = std::chrono::high_resolution_clock::now();
    zed_camera_->fetch_measurements();
    std::vector<tracked_target_info> vec;
//    tag_manager_->detect_tags(zed_camera_, &vec);
    inference_manager_->inference_on_device(zed_camera_);
    zed_camera_->retrieve_objects(objs_);
//    if (image_pub_ != nullptr) {
//        cv::Mat img = slMat_to_cvMat(zed_camera_->get_left_image());
//        cv::Mat img_new;
//        cv::cvtColor(img, img_new, cv::COLOR_BGRA2BGR);
//        debug("Num objs (image_sender): " + std::to_string(objs_.object_list.size()));
//        for (auto& object : objs_.object_list) {
//            cv::Point p1(object.bounding_box_2d.at(0).x, object.bounding_box_2d.at(0).y);
//            cv::Point p2(object.bounding_box_2d.at(2).x, object.bounding_box_2d.at(2).y);
//            cv::Rect r(p1, p2);
//            std::string id = std::to_string(object.id);
//            cv::rectangle(img_new, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//            cv::putText(img_new, id, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
//        }
//        image_pub_->img_ = img_new;
//    }
    if (vision_pub_ != nullptr) {
        objects_to_tracked_target_info(objs_, &vec);
//        debug("Is objects new: " + std::to_string(objs_.is_new));
        vision_pub_->targets_ = vec;
    }
    zmq_manager_->send_publishers();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    debug("Time took: " + std::to_string(elapsed.count()));
}

VisionRunner::~VisionRunner() {
    delete image_pub_;
    delete vision_pub_;
}
