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

    state_estimator_ = new EstimatorContainer<float>(
            control_input_,
            measurements_,
            &state_estimate_
            );

    initialize_state_estimator();
}

void VisionRunner::run() {
    state_estimator_->run();
}

void VisionRunner::initialize_state_estimator() {
    state_estimator_->remove_all_estimators();
    state_estimator_->add_estimator<MCLPoseEstimator<float>>();
}

VisionRunner::~VisionRunner() {
    delete state_estimator_;
}
