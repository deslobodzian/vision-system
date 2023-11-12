//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_VISION_CONTAINER_HPP
#define VISION_SYSTEM_VISION_CONTAINER_HPP

#include "vision_runner.hpp"

class VisionContainer {
public:
    VisionContainer();

    void init();
    void run();

    virtual ~VisionContainer();

protected:
    std::shared_ptr<TaskManager> task_manager_;
    std::shared_ptr<VisionRunner> vision_runner_;
};

#endif //VISION_SYSTEM_VISION_CONTAINER_HPP
