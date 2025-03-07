#ifdef CUDA
#include "zed.h"
#include "logger.h"

ZedCamera::ZedCamera() : tracking_enabled_(false) {
    name_ = "ZED Camera";
}

ZedCamera::~ZedCamera() {
    if (zed_.isOpened()) {
        close();
    }
}

int ZedCamera::open() {
    sl::InitParameters init_params;
    sl::RuntimeParameters runtime_params;
    return open(init_params, runtime_params);
}

bool ZedCamera::is_open() {
    return zed_.isOpened();
}

bool ZedCamera::successful_grab() {  // Fixed typo in function name
    grab_state_ = zed_.grab(runtime_params_);
    return (grab_state_ == sl::ERROR_CODE::SUCCESS);
}

int ZedCamera::fetch_measurements(const MeasurementType& types, const sl::MEM& memory_type) {
    if (successful_grab()) {  // Fixed function name call
        measurements_.timestamp = zed_.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        
        if (has_measurement(types, MeasurementType::IMAGE)) {
            LOG_DEBUG("Fetching Image");
            zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type);
        }

        if (has_measurement(types, MeasurementType::DEPTH)) {
            LOG_DEBUG("Fetching Depth");
            zed_.retrieveMeasure(measurements_.depth_map, sl::MEASURE::DEPTH);
        }

        if (has_measurement(types, MeasurementType::SENSORS)) {
            LOG_DEBUG("Fetching Sensors");
            zed_.getSensorsData(measurements_.sensors_data, sl::TIME_REFERENCE::IMAGE);
        }

        if (has_measurement(types, MeasurementType::OBJECTS)) {
            // object grab
            LOG_ERROR("NOT IMPLEMENTED YET");
        }

        if (has_measurement(types, MeasurementType::POSE)) {
            if (tracking_enabled_) {
                zed_.getPosition(measurements_.camera_pose, sl::REFERENCE_FRAME::WORLD);
            } else {
                LOG_ERROR("Pose tracking is not enabled!");
            }
        }
        return 0;
    }
    return -1;  // Return error code on failed grab
}

const ZedMeasurements& ZedCamera::get_measurements() {
   return measurements_; 
}

int ZedCamera::open(const sl::InitParameters& init_params, const sl::RuntimeParameters& runtime_params) {
    init_params_ = init_params;
    runtime_params_ = runtime_params;

    auto ret = zed_.open(init_params_);
    if (ret != sl::ERROR_CODE::SUCCESS) {
        LOG_ERROR("Failed to open ZED camera with error: ", sl::toVerbose(ret));
        return -1;  // Return error code
    }
    return 0;
}

int ZedCamera::enable_tracking(const sl::PositionalTrackingParameters& tracking_params) {
    auto ret = zed_.enablePositionalTracking(tracking_params);
    if (ret != sl::ERROR_CODE::SUCCESS) {
        LOG_ERROR("Failed to enable tracking with error: ", sl::toVerbose(ret));  // Fixed error message
        tracking_enabled_ = false;
        return -1;
    }
    tracking_enabled_ = true;
    return 0;
}

int ZedCamera::enable_streaming() {
    sl::StreamingParameters stream_params;
    stream_params.codec = sl::STREAMING_CODEC::H264;
    stream_params.bitrate = 8000;
    stream_params.port = 30000;

    auto ret = zed_.enableStreaming(stream_params);
    return (ret == sl::ERROR_CODE::SUCCESS) ? 0 : -1;  // Consistent return values
}

void ZedCamera::disable_streaming() {
    zed_.disableStreaming();
}

void ZedCamera::close() {
    LOG_INFO("Closing Camera");
    zed_.close();
}

std::string ZedCamera::camera_status_string() {
    auto init_params = zed_.getInitParameters();
    auto tracking_state = zed_.getPositionalTrackingStatus();
    std::stringstream ss;
    ss << "CAMERA STATUS: " << "\n"
    << "[SDK VERSION]: " << zed_.getSDKVersion() << "\n"
    << "[OPENED STATE]: " << zed_.isOpened() << "\n"
    << "[LAST GRAB STATE]: " << grab_state_ << "\n" 
    << "[INIT PARAMETERS]: " << "\n"
    << "----[RESOLUTION]: " << init_params.camera_resolution << "\n"
    << "----[FPS]: " << init_params.camera_fps << "\n"
    << "----[CAMERA FLIPED]: " << init_params.camera_image_flip << "\n"
    << "----[DEPTH MODE]: " << init_params.depth_mode << "\n"
    << "----[DEPTH MIN]: " << init_params.depth_minimum_distance<< "\n"
    << "----[DEPTH MAX]: " << init_params.depth_maximum_distance << "\n"
    << "----[COORDINATE SYSTEM]: " << init_params.coordinate_system << "\n"
    << "----[COORDINATE UNITS]: " << init_params.coordinate_units << "\n"
    << "\n"

    << "[RUNNING STATE]: " << "\n"
    << "----[CURRENT FPS]: " << zed_.getCurrentFPS() << "\n"
    << "----[DROPPED FRAMES COUNT]: " << zed_.getFrameDroppedCount() << "\n"
    //<< "----[SPATIAL MAPPING STATUS]: " << zed_.getSpatialMappingState() << "\n"
    << "\n" 

    << "[POSITIONAL TRACKING STATUS]: " <<  "\n"
    << "----[ODOMETRY STATUS]: " << tracking_state.odometry_status << "\n"
    << "----[SPATIAL MEMORY STATUS]: " << tracking_state.spatial_memory_status << "\n"
    << "----[TRACKING FUSION STATUS]: " << tracking_state.tracking_fusion_status << "\n"

    << "\n";
    return ss.str();
}

#endif /* CUDA */
