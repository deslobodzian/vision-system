//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_VISION_CONTAINER_HPP
#define VISION_SYSTEM_VISION_CONTAINER_HPP

#include "april_tag_runner.hpp"
#include "networking/zmq_manager.hpp"
#include "vision_runner.hpp"

class VisionContainer {
public:
  VisionContainer();

  void init();
  void run();

  virtual ~VisionContainer();

protected:
  void zmq_heart_beat();

  std::shared_ptr<TaskManager> task_manager_;
  std::shared_ptr<VisionRunner> vision_runner_;
  std::shared_ptr<AprilTagRunner> april_tag_runner_;
  std::shared_ptr<ZmqManager> zmq_manager_;

#ifdef WITH_CUDA
  std::vector<sl::DeviceProperties> dev_list_;
#endif /* WITH_CUDA */
  std::vector<unsigned int> serial_numbers_;
};

#endif // VISION_SYSTEM_VISION_CONTAINER_HPP
