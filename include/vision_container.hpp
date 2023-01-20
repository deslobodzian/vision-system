//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_VISION_CONTAINER_HPP
#define VISION_SYSTEM_VISION_CONTAINER_HPP


#include "estimator/estimator.hpp"
#include "utils/task.hpp"
#include "utils/utils.hpp"
#include "vision_runner.hpp"
#include "vision/Zed.hpp"
#include "vision/monocular_camera.hpp"
#include "vision/apriltag_manager.hpp"
#include "inference/inference_manager.hpp"
#include "networking/zmq_manager.hpp"

class VisionContainer {
public:
    VisionContainer();

    void init();
    void run();

    virtual ~VisionContainer();

protected:
    TaskManager task_manager_;
    ZmqManager* zmq_manager_ = nullptr;
    InferenceManager* inference_manager_ = nullptr;
    Zed* zed_camera_ = nullptr;
//    AprilTagManager<float>* tag_manager_ = nullptr;
    VisionRunner* vision_runner_ = nullptr;

    void detect_zed_targets();
//    void detect_monocular_targets();
};

#endif //VISION_SYSTEM_VISION_CONTAINER_HPP
