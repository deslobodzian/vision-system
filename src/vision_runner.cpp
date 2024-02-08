//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"
#include "utils/logger.hpp"
#include "utils/timer.h"
#include "vision_pose_generated.h"
#include "vision_pose_array_generated.h"

VisionRunner::VisionRunner(
        std::shared_ptr<TaskManager> manager,
        double period,
        const std::string& name,
        const std::shared_ptr<ZmqManager> zmq_manager) :
        Task(manager, period, name),
        zmq_manager_(zmq_manager)
        {
#ifdef WITH_CUDA
    cfg_.res = sl::RESOLUTION::SVGA;
    //cfg_.res = sl::RESOLUTION::VGA;
    cfg_.sdk_verbose = true;
    cfg_.enable_tracking = true;
    cfg_.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    cfg_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    cfg_.max_depth = 20;
    cfg_.prediction_timeout_s = 0.2f;
    cfg_.batch_latency = 0.2f;
    cfg_.id_retention_time = 0.f;
    cfg_.detection_confidence_threshold = 50;
    cfg_.default_memory = sl::MEM::GPU;
    camera_.configure(cfg_);
    camera_.open();
#endif
}

void VisionRunner::init() {
    LOG_INFO("Initializing [VisionRunner]");
#ifdef WITH_CUDA
    camera_.enable_tracking();
    camera_.enable_object_detection();
    detection_config det_cfg;
    det_cfg.nms_thres = 0.5;
    det_cfg.obj_thres = 0.5;
    detector_.configure(det_cfg);
#endif
}

void VisionRunner::run() {
    Timer t;
#ifdef WITH_CUDA 
    camera_.fetch_measurements(MeasurementType::IMAGE_AND_OBJECTS);
    detector_.detect_objects(camera_);
    const sl::Objects& objects = camera_.retrieve_objects();
    LOG_DEBUG("Detected Objects: ", objects.object_list.size());
        std::vector<flatbuffers::Offset<Messages::VisionPose>> visionPoseOffsets;

    auto& builder = zmq_manager_->get_publisher("main").get_builder(); 

    for (const auto& obj : objects.object_list) {
        auto visionPose = Messages::CreateVisionPose(
            builder, 
            obj.id, 
            obj.position.x, 
            obj.position.y, 
            obj.position.z, 
            t.get_ms() 
        );
        visionPoseOffsets.push_back(visionPose);
    }

    auto posesVector = builder.CreateVector(visionPoseOffsets);

    zmq_manager_->get_publisher("main").publish(
        "VisionPoseArray", 
        Messages::CreateVisionPoseArray,
        posesVector 
    );
#endif
    zmq_manager_->get_publisher("main").publish(
            "VisionPose",
            Messages::CreateVisionPose,
            123,     1.0f, 2.0f, 3.0f, 0.0
    );
}

VisionRunner::~VisionRunner() {

}
