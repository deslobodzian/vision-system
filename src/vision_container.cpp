//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"

VisionContainer::VisionContainer() {}

void VisionContainer::init_scheduler() {
    info("[Scheduler]: Configuring real time priority.");
    struct sched_param parameters_{};
    parameters_.sched_priority = PRIORITY;
    if (sched_setscheduler(0, SCHED_FIFO, &parameters_) == -1) {
        error("[Scheduler]: Failed to configure task scheduler.");
    }
}
void VisionContainer::init() {
    info("[VisionContainer]: Initialize Scheduler");
    init_scheduler();

    info("[VisionContainer]: Subscribing to odometry");
    nt_manager_.add_subscriber(&odometry_sub_);
}

//void VisionContainer::run() {
//    init();
//
//    PeriodicMemberFunction<VisionContainer> nt_task(
//            &task_manager
//            )
//}