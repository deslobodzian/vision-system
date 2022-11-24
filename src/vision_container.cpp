//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"

VisionContainer::VisionContainer() :

void VisionContainer::init_scheduler() {
    printf("[Scheduler]: Configuring real time scheduler priority. \n");
    struct sched_param parameters_{};
    parameters_.sched_priority = PRIORITY;
    if (sched_setscheduler(0, SCHED_FIFO, &parameters_) == -1) {
        printf("[ERROR]: Failed to configure task scheduler. \n");
    }
}
//void VisionContainer::init() {
//    printf("[VisionContainer]: Initialize Scheduler");
//    init_scheduler();
//}
//
//void VisionContainer::run() {
//    init();
//
//    PeriodicMemberFunction<VisionContainer> nt_task(
//            &task_manager
//            )
//}