//
// Created by deslobodzian on 11/23/22.
//
#include "vision_runner.hpp"
#include "utils/logger.hpp"
#include "utils/april_tag_utils.hpp"

VisionRunner::VisionRunner(
        std::shared_ptr<TaskManager> manager,
        double period,
        const std::string& name,
        const std::shared_ptr<ZmqManager> zmq_manager) :
        Task(manager, period, name),
        zmq_manager_(zmq_manager),
        use_detection_(false)
        {
#ifdef WITH_CUDA
    // Dennis's camera: 47502321
    // Outliers's camera: 41535987
    cfg_.serial_number = 41535987;
   	// cfg_.serial_number = 47502321;
    cfg_.res = sl::RESOLUTION::SVGA;
    //cfg_.res = sl::RESOLUTION::VGA;
    cfg_.sdk_verbose = false;
    cfg_.enable_tracking = false; // object detection
    cfg_.depth_mode = sl::DEPTH_MODE::ULTRA; 
    //cfg_.depth_mode = sl::DEPTH_MODE::PERFORMANCE; 
    cfg_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    cfg_.max_depth = 5;
    cfg_.min_depth = 0.3f;

    cfg_.async_grab = false;

    cfg_.prediction_timeout_s = 0.04f; // 40ms 
    cfg_.batch_latency = 0.010f;  // 10ms
    cfg_.id_retention_time = 0.04f; // 40ms
    cfg_.detection_confidence_threshold = 50;
    cfg_.default_memory = sl::MEM::GPU;

    sl::Resolution res = sl::getResolution(cfg_.res);
    res.height = res.height / 2;
    res.width = res.width / 2;

    cfg_.depth_resolution = res;
    camera_.configure(cfg_);
    camera_.open();

    const sl::Resolution display_resolution = camera_.get_resolution();
    const uint32_t img_height = display_resolution.height;
    const uint32_t img_width = display_resolution.width;
    constexpr uint32_t tile_size = 4; 
    constexpr cuAprilTagsFamily tag_family = NVAT_TAG36H11; 
    constexpr float tag_dim = 0.16f;
    constexpr int decimate = 2;
    //tag_detector_.init_detector(img_width, img_height, tile_size, tag_family, tag_dim, decimate);
#endif
}

void VisionRunner::init() {
    LOG_INFO("Initializing [VisionRunner]");
#ifdef WITH_CUDA
    //camera_.enable_tracking();
    camera_.enable_object_detection();
    detection_config det_cfg;
    det_cfg.nms_thres = 0.5;
    det_cfg.obj_thres = 0.5;
    detector_.configure(det_cfg);
#endif
}

void VisionRunner::run() {
    using namespace std::chrono;
#ifdef WITH_CUDA 
    const auto start_time = high_resolution_clock::now();
    auto& builder = zmq_manager_->get_publisher("BackZed").get_builder(); 

    LOG_DEBUG("Fetching measurement IMAGE");
    camera_.fetch_measurements(MeasurementType::IMAGE); // use for async grab
    LOG_DEBUG("Detect objects");
    detector_.detect_objects(camera_);
    //        camera_.fetch_measurements(MeasurementType::OBJECTS);
    camera_.fetch_objects();
    const sl::Objects& objects = camera_.retrieve_objects();

    const auto object_detection_time = high_resolution_clock::now();
    const auto object_detection_ms = duration_cast<milliseconds>(object_detection_time - start_time).count();
    LOG_DEBUG("Object detection took: ", object_detection_ms, " ms");

    std::vector<flatbuffers::Offset<Messages::VisionPose>> vision_pose_offsets;
    vision_pose_offsets.reserve(objects.object_list.size());

    for (const auto& obj : objects.object_list) {
	    LOG_DEBUG("Object: [", obj.id, ", ", obj.position.x, ", ", obj.position.y, ", ", obj.position.z, "]");
//	    if (obj.tracking_state == sl::OBJECT_TRACKING_STATE::OK) {
		    auto vision_pose = Messages::CreateVisionPose(
				    builder, 
				    obj.id,
				    obj.position.x, 
				    obj.position.y, 
				    obj.position.z, 
				    object_detection_ms
				    );
		    vision_pose_offsets.push_back(vision_pose);
//	    }
    }
    auto poses_vector = builder.CreateVector(vision_pose_offsets);
    auto vision_pose_array = Messages::CreateVisionPoseArray(builder, poses_vector);

    builder.Finish(vision_pose_array);
    LOG_DEBUG("Builder size: ", builder.GetSize());
    zmq_manager_->get_publisher("BackZed").publish_prebuilt(
            "Objects",
            builder.GetBufferPointer(),
            builder.GetSize()  
            );
    const auto pipeline = high_resolution_clock::now();
    const auto pipeline_ms= duration_cast<milliseconds>(pipeline - start_time).count();
    LOG_DEBUG("Zed Pipeline took: ", pipeline_ms, " ms");
#endif /* WITH_CUDA */
    LOG_DEBUG("Not using CUDA");
}

VisionRunner::~VisionRunner() {
	camera_.close();
}
