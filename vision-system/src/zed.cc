#include "zed.h"
#include "logger.h"

ZedCamera::ZedCamera() {

}

bool ZedCamera::sucessfull_grab()  {
    grab_state_ = zed_.grab();
    return (grab_state_ == ERROR_CODE::SUCCESS);
}

int ZedCamera::fetch_measurements(const MeasurementType& types, const sl::MEM& memory_type) {
    if (sucessfull_grab()) {
        if (has_measurement(types, MeasurementType::IMAGE)) {
            LOG_DEBUG("Fetching Image");
            zed_.retrieveImage(measurements_.left_image, VIEW::LEFT, memory_type);
        }

        if (has_measurement(types, MeasurementType::DEPTH)) {
            LOG_DEBUG("Fetching Depth");
            zed_.retrieveMeasure(measurements_.depth_map, MEASURE::DEPTH);
        }

        if (has_measurement(types, MeasurementType::SENSORS)) {
            LOG_DEBUG("Fetching Sensors");
            zed_.getSensorsData(measurements_.sensors_data, TIME_REFERENCE::IMAGE);
        }

        if (has_measurement(types, MeasurementType::OBJECTS)) {
            // object grab
            LOG_ERROR("NOT IMPLEMENTED YET");
        }
        return 0;
    }
    return 0;
}


int ZedCamera::open(const InitParameters& init_params) {
    init_params_ = init_params;

    auto ret = zed_.open(init_params_);
    if (ret != ERROR_CODE::SUCCESS) {
        LOG_ERROR("Failed to open ZED camera with error: ", sl::toVerbose(ret));
    }
    return 0;
}

std::string ZedCamera::camera_status_string() {
    auto init_params = zed_.getInitParameters();
    auto tracking_state = zed_.getPositionalTrackingStatus();
    std::stringstream ss;
    ss << "CAMERA STATUS: " << "\n" 
    << "[SDK VERSION]: " << zed_.getSDKVersion() << "\n"
    << "[OPENED STATE]: " << zed_.isOpened() << "\n"
    << "[LAST GRAB STATE]" << grab_state_ << "\n" 
    << "[INIT PARAMETERS]" << "\n"
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
    << "----[SPATIAL MAPPING STATUS]: " << zed_.getSpatialMappingState() << "\n"
    << "\n" 

    << "[POSITIONAL TRACKING STATUS]: " <<  "\n"
    << "----[ODOMETRY STATUS]: " << tracking_state.odometry_status << "\n"
    << "----[SPATIAL MEMORY STATUS]: " << tracking_state.spatial_memory_status << "\n"
    << "----[TRACKING FUSION STATUS]: " << tracking_state.tracking_fusion_status << "\n"

    << "\n";
    return ss.str();
}

