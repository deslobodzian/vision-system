//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"

VisionRunner::VisionRunner(
        std::shared_ptr<TaskManager> manager,
        double period,
        const std::string& name):
        Task(manager, period, name) {
}

void VisionRunner::init() {
    // info("Initializing [VisionRunner]");
}

void VisionRunner::run() {
}

VisionRunner::~VisionRunner() {
}
