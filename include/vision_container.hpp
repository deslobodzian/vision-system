//
// Created by deslobodzian on 11/22/22.
//

#ifndef VISION_SYSTEM_VISION_CONTAINER_HPP
#define VISION_SYSTEM_VISION_CONTAINER_HPP

#define PRIORITY 49

#include "estimator/estimator.hpp"
#include <sys/mman.h>
#include "utils/task.hpp"
#include "utils/utils.hpp"
#include "networking/odometry_sub.hpp"
#include "networking/vision_pub.hpp"
#include "networking/nt_manager.hpp"

class VisionContainer {
public:
    VisionContainer();

    void init_scheduler();
    void init();

    virtual ~VisionContainer();

    std::vector<Measurement<double>>* measurements_;

    void get_odometry_data();

protected:
    TaskManager task_manager_;
    NTManager nt_manager_;

    odometry_subscribable odometry_sub_;


};

#endif //VISION_SYSTEM_VISION_CONTAINER_HPP
