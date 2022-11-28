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
//    nt_manager_.add_subscriber(&odometry_sub_);
    info("[VisionContainer]: Starting odometry subscription thread");
//    odometry_thread_ = std::thread(&VisionContainer::odometry_handle, this);

    info("[VisionContainer]: Setting up zed camera");
    zed_config zed_config{};
    zed_config.res = sl::RESOLUTION::VGA;
    zed_config.fps = 100;
    zed_config.depth_mode = sl::DEPTH_MODE::ULTRA;
    zed_config.sdk_verbose = false;
    zed_config.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    zed_config.units = sl::UNIT::METER;
    zed_config.max_depth = 20.0;

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

    info("[VisionContainer]: Setting up AprilTag manager");
    detector_config apriltag_config {};
    apriltag_config.tf = tag16h5;
    apriltag_config.quad_decimate = 0.5;
    apriltag_config.quad_sigma = 0.5;
    apriltag_config.nthreads = 2;
    apriltag_config.debug = false;
    apriltag_config.refine_edges = false;

    tag_manager_ = new AprilTagManager(apriltag_config);
}
void VisionContainer::detect_targets() {
    tag_manager_->detect_tags_zed(zed_camera_);
    tag_manager_->detect_tags_monocular(monocular_camera_);
}

void VisionContainer::run() {
    init();
    info("[VisionContainer]: Starting system");

    vision_runner_ = new VisionRunner(&task_manager_, 0.05, "vision-runner");

    vision_runner_->control_input_ = control_input_;
    vision_runner_->measurements_ = measurements_;

    // init threads;
    info("[VisionContainer]: Starting detection task");
    PeriodicMemberFunction<VisionContainer> detection_task(
            &task_manager_,
            0.002,
            "detection",
            &VisionContainer::detect_targets,
            this
            );
    detection_task.start();

    for (;;) {
        usleep(1000000);
    }
}

void VisionContainer::odometry_handle() {
//    nt_manager_.get_subscription(&odometry_sub_);
//    control_input_->set_odometry_input(&odometry_sub_);
}

VisionContainer::~VisionContainer() {
    delete vision_runner_;
    delete tag_manager_;
    delete monocular_camera_;
    delete zed_camera_;
}
