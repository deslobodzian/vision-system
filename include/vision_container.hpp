//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_VISION_CONTAINER_HPP
#define VISION_SYSTEM_VISION_CONTAINER_HPP


#include "estimator/estimator.hpp"
#include "utils/task.hpp"
#include "utils/utils.hpp"
//#include "networking/odometry_sub.hpp"
//#include "networking/vision_pub.hpp"
//#include "networking/nt_manager.hpp"
#include "vision_runner.hpp"
#include "vision/Zed.hpp"
#include "vision/monocular_camera.hpp"
#include "vision/apriltag_manager.hpp"

class VisionContainer {
public:
    VisionContainer();

    void init();
    void run();

    virtual ~VisionContainer();

    std::vector<Measurement<float>>* measurements_;
    ControlInput<float>* control_input_;

    void get_odometry_data();

protected:
    TaskManager task_manager_;
//    NTManager nt_manager_;
    std::thread odometry_thread_;
//    odometry_subscribable odometry_sub_;

    Zed* zed_camera_ = nullptr;
    MonocularCamera<float>* monocular_camera_ = nullptr;
    AprilTagManager<float>* tag_manager_ = nullptr;
    VisionRunner* vision_runner_ = nullptr;

    void odometry_handle();
    void detect_targets();
};

#endif //VISION_SYSTEM_VISION_CONTAINER_HPP
