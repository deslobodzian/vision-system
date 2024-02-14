//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"
#include "heart_beat_generated.h"
#include "networking/zmq_publisher.hpp"
#include "utils/logger.hpp"
#include "utils/task.hpp"
#include "vision/zed.hpp"
#include <flatbuffers/flatbuffer_builder.h>
#include "vision/apriltag_detector.hpp"

VisionContainer::VisionContainer()
    : vision_runner_(nullptr), 
    task_manager_(std::make_shared<TaskManager>()), zmq_manager_(std::make_shared<ZmqManager>()) {
        zmq_manager_->create_publisher("main", "tcp://*:5556");
    }

void drawBoundingBoxes(cv::Mat &image, const std::vector<BBoxInfo> &bboxes) {
    for (const auto &bbox : bboxes) {
        cv::rectangle(image, cv::Point(bbox.box.x1, bbox.box.y1),
                cv::Point(bbox.box.x2, bbox.box.y2), cv::Scalar(0, 255, 0),
                2);
        std::string label = "Class " + std::to_string(bbox.label) + " : " +
            std::to_string(bbox.probability);
        int baseline;
        cv::Size labelSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(
                image,
                cv::Point(bbox.box.x1, bbox.box.y1 - labelSize.height - baseline),
                cv::Point(bbox.box.x1 + labelSize.width, bbox.box.y1),
                cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(bbox.box.x1, bbox.box.y1 - baseline),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void VisionContainer::zmq_heart_beat() {
    using namespace std::chrono;
    auto now = duration_cast<microseconds>(system_clock::now().time_since_epoch())
        .count();
    zmq_manager_->get_publisher("main").publish(
            "HeartBeat", Messages::CreateHeartBeat, 111, now);
}

void VisionContainer::init() {
    LOG_INFO("Init Vision container called");
}

void VisionContainer::run() {
    using namespace std::chrono_literals;
    using namespace std::chrono;
    init();
    vision_runner_ = task_manager_->create_task<VisionRunner>(
            0.01,
            "vision-runner",
            zmq_manager_
            );
    vision_runner_->start();

    LOG_INFO("Starting Heart Beat task");
    PeriodicMemberFunction<VisionContainer> heart_beat(
            task_manager_, 1.0, "heart_beat",
            this,                            // object pointer
            &VisionContainer::zmq_heart_beat // member function pointer
            );
    heart_beat.start();

    for (;;) {
        std::this_thread::sleep_for(100s);
    }
}

VisionContainer::~VisionContainer() {}
