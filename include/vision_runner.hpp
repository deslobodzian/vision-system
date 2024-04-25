//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_VISION_RUNNER_HPP
#define VISION_SYSTEM_VISION_RUNNER_HPP

#include "networking/zmq_manager.hpp"
#include "utils/task.hpp"
#include "vision/apriltag_detector.hpp"
#include "vision/object_detector.hpp"

class VisionRunner : public Task {
 public:
  VisionRunner(std::shared_ptr<TaskManager>, double period, const std::string&,
               const std::shared_ptr<ZmqManager> zmq_manager);
  using Task::Task;
  void init() override;
  void run() override;
  virtual ~VisionRunner();

 private:
#ifdef WITH_CUDA
  zed_config cfg_;
  ZedCamera camera_;
  ObjectDetector<ZedCamera> detector_;
  ApriltagDetector tag_detector_;
#else
  MonocularCamera camera_;
  ObjectDetector<MonocularCamera> detector_;
#endif
  std::shared_ptr<ZmqManager> zmq_manager_;
  bool use_detection_;
};

#endif  // VISION_SYSTEM_VISION_RUNNER_HPP
