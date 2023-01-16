//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"

VisionRunner::VisionRunner(
        TaskManager* manager,
        double period,
        const std::string &name):
        Task(manager, period, name) {
}

void VisionRunner::init() {
    info("initializing [VisionRunner]");
}

void VisionRunner::run() {
}

VisionRunner::~VisionRunner() {
}
