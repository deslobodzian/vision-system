//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"

VisionContainer::VisionContainer() {}

void VisionContainer::init() {
    info("[VisionContainer]: Starting Zmq Manager");
    zmq_manager_ = new ZmqManager();

    info("[VisionContainer]: Setting up zed camera");
    zed_config zed_config{};
    zed_config.res = sl::RESOLUTION::VGA;
    zed_config.fps = 100;
    zed_config.flip_camera = sl::FLIP_MODE::OFF;
    zed_config.depth_mode = sl::DEPTH_MODE::ULTRA;
    zed_config.sdk_verbose = true;
    zed_config.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    zed_config.units = sl::UNIT::METER;
    zed_config.max_depth = 20.0;
    zed_config.reference_frame = REFERENCE_FRAME::CAMERA;
    zed_config.enable_tracking = true;
    zed_config.prediction_timeout_s = 0.2f;
    zed_config.enable_segmentation = false;
    zed_config.enable_batch = false;
    zed_config.batch_latency = 2.f;
    zed_config.id_retention_time = 240.f;
    zed_config.model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    zed_config.detection_confidence_threshold = 50;

    zed_camera_ = new Zed(zed_config);
    // need to open camera before inference manager;
    zed_camera_->open_camera();
    zed_camera_->enable_tracking();
    zed_camera_->enable_object_detection();

    info("[Vision Container]: Starting Inference Manager");
//    inference_manager_ = new InferenceManager("/home/team5687/VisionSystem/engines/jetson_comp.engine");
    inference_manager_ = new InferenceManager("../engines/laptop.engine");
    inference_manager_->init();

//    info("[Vision Container]: Creating IMU publisher");
//    imu_pub_ = new imu_publishable;
//    zmq_manager_->create_publisher(imu_pub_, "tcp://10.56.87.20:5558");
//    std::string folderpath = "/home/prometheus/Projects/VisionSystem/images/*.png"; //images is the folder where the images are stored
//    std::vector<std::string> filenames;
//    cv::glob(folderpath, filenames);
//
//    for (size_t i=0; i<filenames.size(); i++) {
//        cv::Mat im = cv::imread(filenames[i]);
//        inference_manager_->test_inference(im);
//        std::string name = "output" + std::to_string(i)+".jpg";
//        cv::imwrite(name.c_str(), im);
//    }


//    info("[VisionContainer]: Setting up AprilTag manager");
//    detector_config apriltag_config {};
//    apriltag_config.tf = tag16h5;
//    apriltag_config.quad_decimate = 1;
//    apriltag_config.quad_sigma = 0.1;
//    apriltag_config.nthreads = 8;
//    apriltag_config.debug = false;
//    apriltag_config.refine_edges = true;
//    apriltag_config.decision_margin = 40.0f;
//
//    tag_manager_ = new AprilTagManager<float>(apriltag_config);
}

//void VisionContainer::detect_april_tags() {
//    zed_camera_->fetch_measurements();
//    tag_manager_->detect_tags();
//}

void VisionContainer::zmq_publish() {
    if (zmq_manager_ != nullptr) {
        zmq_manager_->send_publishers();
    }
}
void VisionContainer::read_imu() {
    zed_camera_->fetch_measurements();
    sl::float3 imu_orientation = zed_camera_->get_imu_data().pose.getEulerAngles();
//    info("Yaw: " + std::to_string(imu_orientation.z));
    if (imu_pub_ != nullptr) {
        imu_pub_->imu_data_[0] = imu_orientation.z;
        imu_pub_->imu_data_[1] = imu_orientation.y;
        imu_pub_->imu_data_[2] = imu_orientation.x;
    }
//    error("IMUPub data: " +
//        std::to_string(imu_pub_->imu_data_[0])
//        + ", " +
//        std::to_string(imu_pub_->imu_data_[1])
//        + ", " +
//        std::to_string(imu_pub_->imu_data_[2])
//    );
}


void VisionContainer::run() {
    init();
    info("[VisionContainer]: Starting system");
    vision_runner_ = new VisionRunner(&task_manager_, 0.025, "vision-runner");
    vision_runner_->zed_camera_ = zed_camera_;
    vision_runner_->inference_manager_ = inference_manager_;
    vision_runner_->zmq_manager_ = zmq_manager_;
//    vision_runner_->tag_manager_ = tag_manager_;
    vision_runner_->start();

//    info("[VisionContainer]: Starting IMU reading task");
//    PeriodicMemberFunction<VisionContainer> imu_task(
//            &task_manager_,
//            0.01,
//            "imu",
//            &VisionContainer::read_imu,
//            this
//    );
//    imu_task.start();

//    info("[VisionContainer]: Starting publisher task");
//    PeriodicMemberFunction<VisionContainer> publisher_task(
//            &task_manager_,
//            0.001,
//            "imu",
//            &VisionContainer::zmq_publish,
//            this
//    );
//    publisher_task.start();
//    info("[VisionContainer]: Publisher task started");

    for (;;) {
        usleep(1000000);
    }
}

VisionContainer::~VisionContainer() {
    delete vision_runner_;
    delete zed_camera_;
    delete inference_manager_;
    delete zmq_manager_;
    delete imu_pub_;
}
