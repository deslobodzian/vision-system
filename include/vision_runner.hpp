//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_VISION_RUNNER_HPP
#define VISION_SYSTEM_VISION_RUNNER_HPP

#include "utils/task.hpp"
#include "estimator/estimator.hpp"
#include "estimator/mcl_pose_estimator.hpp"

class VisionRunner : public Task {
public:
    VisionRunner(TaskManager*, double, const std::string&);
    using Task::Task;
    void init() override;
    void run() override;

    virtual ~VisionRunner();

private:

};

#endif //VISION_SYSTEM_VISION_RUNNER_HPP
