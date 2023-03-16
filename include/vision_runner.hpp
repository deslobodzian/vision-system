//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_VISION_RUNNER_HPP
#define VISION_SYSTEM_VISION_RUNNER_HPP

#include "utils/task.hpp"
#include "estimator/estimator.hpp"
#include "vision/apriltag_manager.hpp"
#include "estimator/mcl_pose_estimator.hpp"
#include "inference/inference_manager.hpp"
#include "networking/zmq_manager.hpp"
#include "networking/image_pub.hpp"
#include "networking/vision_pub.hpp"

class VisionRunner : public Task {
public:
    VisionRunner(TaskManager*, double, const std::string&);
    using Task::Task;
    void init() override;
    void run() override;

    Zed* zed_camera_;
    InferenceManager* inference_manager_;
    ZmqManager* zmq_manager_;
    AprilTagManager<float>* tag_manager_;

    virtual ~VisionRunner();

private:
    image_publishable* image_pub_ = nullptr;
    vision_publishable* vision_pub_ = nullptr;
    sl::Objects objs_;
};

#endif //VISION_SYSTEM_VISION_RUNNER_HPP
