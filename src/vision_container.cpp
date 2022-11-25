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

    info("[VisionContainer]: Setting up zed camera");
    zed_config zed_config{
        sl::RESOLUTION::VGA,
        100,
        sl::DEPTH_MODE::ULTRA,
        false,
        sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD,
        sl::UNIT::METER,
        20.0
    };
    zed_camera_ = new Zed(zed_config);

    info("[VisionContainer]: Setting up monocular cameras");
    IntrinsicParameters<float> parameters{1116.821, 1113.573, 678.58, 367.73};
    CameraConfig<float> monocular_config (
            "/dev/video0", // need to find a better way for device id.
            68.5,
            resolution(320, 240),
            30,
            parameters
    );
    monocular_camera_ = new MonocularCamera<float>(monocular_config);

}

void VisionContainer::run() {
    init();
    info("[VisionContainer]: Starting system");

    vision_runner_ = new VisionRunner(&task_manager_, 0.05, "vision-runner");

    vision_runner_->control_input_ = control_input_;
    vision_runner_->measurements_ = measurements_;

}