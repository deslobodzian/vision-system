//
// Created by deslobodzian on 11/23/22.
//
#include "april_tag_runner.hpp"
#include "utils/logger.hpp"
#include "utils/timer.h"
#include "utils/zmq_flatbuffers_utils.hpp"
#include "vision_pose_generated.h"
#include "vision_pose_array_generated.h"
#include "networking/zmq_subscriber.hpp"

AprilTagRunner::AprilTagRunner(
        std::shared_ptr<TaskManager> manager,
        double period,
        const std::string& name,
        const std::shared_ptr<ZmqManager> zmq_manager) :
        Task(manager, period, name),
        zmq_manager_(zmq_manager)
        {
#ifdef WITH_CUDA
    // Dennis's camera: 47502321
    // Outliers's camera: 41535987
    cfg_.serial_number = 47502321;
    cfg_.res = sl::RESOLUTION::SVGA;
    //cfg_.res = sl::RESOLUTION::VGA;
    cfg_.sdk_verbose = true;
    cfg_.enable_tracking = true;
    cfg_.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    cfg_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    cfg_.max_depth = 20;
    cfg_.default_memory = sl::MEM::GPU;
    camera_.configure(cfg_);
    camera_.open();

    sl::Resolution display_resolution = camera_.get_resolution();

    uint32_t img_height = display_resolution.height;
    uint32_t img_width = display_resolution.width;
    uint32_t tile_size = 4; 
    cuAprilTagsFamily tag_family = NVAT_TAG36H11; 
    float tag_dim = 0.16f;
    tag_detector_.init_detector(img_width, img_height, tile_size, tag_family, tag_dim);
#endif
}

void AprilTagRunner::init() {
    LOG_INFO("Initializing [AprilTagRunner]");
}

void AprilTagRunner::run() {
    Timer t;
    t.start();
    bool use_detection;

    auto received = zmq_manager_->get_subscriber("UseDetection").receive();
    const auto& [topic, msg] = *received;
    if (received) {
            const auto& [topic, msg] = *received;
            if (topic == "UseDetection") { 
                use_detection = process_use_detection(msg);
            }
    }
    // Use this to bypass message
    //use_detectinons = false;

#ifdef WITH_CUDA 
    if (!use_detection) {
        LOG_DEBUG("Using apriltag detection");
        camera_.fetch_measurements(MeasurementType::IMAGE_AND_POINT_CLOUD);
        auto tags = tag_detector_.detect_april_tags_in_sl_image(camera_.get_left_image(), camera_.get_cuda_stream());
        auto zed_tags = tag_detector_.calculate_zed_apriltag(camera_.get_point_cloud(), tags);
        std::vector<flatbuffers::Offset<Messages::VisionPose>> vision_pose_offsets;

        auto& builder = zmq_manager_->get_publisher("main").get_builder(); 

        auto current_ms  = t.get_ms();
        for (const auto& tag : zed_tags) {
            auto vision_pose = Messages::CreateVisionPose(
                    builder,
                    tag.tag_id,
                    tag.center.x,
                    tag.center.y,
                    tag.center.z,
                    current_ms // now just how long processing takes for latency (will roughly be 20ms)
                    );
            vision_pose_offsets.push_back(vision_pose);
        }

        auto poses_vector = builder.CreateVector(vision_pose_offsets);
        auto vision_pose_array = Messages::CreateVisionPoseArray(builder, poses_vector);

        zmq_manager_->get_publisher("main").publish_prebuilt(
                "AprilTags", 
                builder.GetBufferPointer(),
                builder.GetSize()  
                );
        builder.Finish(vision_pose_array);
        current_ms  = t.get_ms();
        LOG_DEBUG("Zed pipline took: ", current_ms, " ms");
    } else {
        LOG_DEBUG("Using notes detection only not running tags to save resources");
    }
#endif
}

AprilTagRunner::~AprilTagRunner() {

}
