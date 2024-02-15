//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_APRIL_TAG_RUNNER_HPP
#define VISION_SYSTEM_APRIL_TAG_RUNNER_HPP

#include "utils/task.hpp"
#include "networking/zmq_manager.hpp"
#include "vision/apriltag_detector.hpp"

class AprilTagRunner : public Task {
public:
    AprilTagRunner(
            std::shared_ptr<TaskManager>,
            double period,
            const std::string&,
            const std::shared_ptr<ZmqManager> zmq_manager
            );
    using Task::Task;
    void init() override;
    void run() override;
    virtual ~AprilTagRunner();
private:
#ifdef WITH_CUDA
    zed_config cfg_;
    ZedCamera camera_;
    ApriltagDetector tag_detector_;
#endif
    std::shared_ptr<ZmqManager> zmq_manager_;

};

#endif /* VISION_SYSTEM_APRIL_TAG_RUNNER_HPP */
