//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"
#include "heart_beat_generated.h"
#include "networking/zmq_publisher.hpp"
#include "utils/logger.hpp"
#include "utils/task.hpp"
#include "vision/apriltag_detector.hpp"
#include "vision/zed.hpp"
#include <algorithm>
#include <flatbuffers/flatbuffer_builder.h>

VisionContainer::VisionContainer()
    : vision_runner_(nullptr),
      april_tag_runner_(nullptr),
      task_manager_(std::make_shared<TaskManager>()),
      zmq_manager_(std::make_shared<ZmqManager>()) {
  //  zmq_manager_->create_publisher("FrontZed", "tcp://*:5556");
  zmq_manager_->create_publisher("BackZed", "ipc:///home/orin/tmp/vision/0");
  // zmq_manager_->create_publisher("BackZed", "tcp://10.56.87.20:5556");
  //   zmq_manager_->create_subscriber("UseDetection", "tcp://localhost:5558");
}

void drawBoundingBoxes(cv::Mat& image, const std::vector<BBoxInfo>& bboxes) {
  for (const auto& bbox : bboxes) {
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
  // auto now =
  // duration_cast<microseconds>(system_clock::now().time_since_epoch())
  //                .count();
  // zmq_manager_->get_publisher("main").publish(
  //     "HeartBeat", Messages::CreateHeartBeat, 111, now);
}

void VisionContainer::init() {
  using namespace std::chrono_literals;
  using namespace std::chrono;
  LOG_INFO("Init Vision container called");
  int retries = 0;

#ifdef WITH_CUDA
  dev_list_ = sl::Camera::getDeviceList();
  int nb_detected_zed = dev_list_.size();

  while (retries <= 5) {
    dev_list_ = sl::Camera::getDeviceList();
    nb_detected_zed = dev_list_.size();

    if (nb_detected_zed <= 0) {
      LOG_ERROR("Zed camera not detected, retrying after a second!");
      retries++;
      std::this_thread::sleep_for(1s);
    } else {
      bool camera_available = true;
      for (int z = 0; z < nb_detected_zed; z++) {
        if (dev_list_[z].camera_state == sl::CAMERA_STATE::NOT_AVAILABLE) {
          LOG_ERROR("Zed camara not available restarting zed daemon!");
          camera_available = false;
          break;
        }
      }
      if (!camera_available) {
        auto ret = std::system("systemctl restart zed_x_daemon.service");
        LOG_ERROR("zed daemon reset returned ", ret);
        LOG_ERROR("Restarting Program");
        break;
      } else {
        break;
      }
    }
  }

  for (int z = 0; z < nb_detected_zed; z++) {
    std::cout << "ID : " << dev_list_[z].id
              << " ,model : " << dev_list_[z].camera_model
              << " , S/N : " << dev_list_[z].serial_number
              << " , state : " << dev_list_[z].camera_state << std::endl;
    serial_numbers_.push_back(dev_list_[z].serial_number);
  }
#endif /* WITH_CUDA */
}

void VisionContainer::run() {
  using namespace std::chrono_literals;
  using namespace std::chrono;

  const static unsigned int VISION_RUNNER_SN = 41535987;
  // const static unsigned int VISION_RUNNER_SN = 47502321;
  const static unsigned int APRIL_TAG_RUNNER_SN = 47502321;
  // const static unsigned int APRIL_TAG_RUNNER_SN = 41535987;
  init();
  auto has_vision_runner = std::find(serial_numbers_.begin(),
                                     serial_numbers_.end(), VISION_RUNNER_SN);

  if (has_vision_runner != serial_numbers_.end()) {
    LOG_INFO("Vision Runner camera detected, starting task");
    vision_runner_ = task_manager_->create_task<VisionRunner>(
        0.015, "vision-runner", zmq_manager_);
    vision_runner_->start();
  } else {
    LOG_ERROR(
        "No Vision Runner camera found, task not starting and restating!");
    return;
  }

  for (;;) {
    std::this_thread::sleep_for(100s);
  }
}

VisionContainer::~VisionContainer() {}
