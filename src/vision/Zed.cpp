//
// Created by DSlobodzian on 4/21/2022.
//
#include "vision/Zed.hpp"

Zed::Zed() {
    // Initial Parameters
    init_params_.camera_resolution = RESOLUTION::VGA;
    init_params_.camera_fps = 100;
    init_params_.depth_mode = DEPTH_MODE::ULTRA;
    init_params_.sdk_verbose = true;
    init_params_.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params_.coordinate_units = UNIT::METER;
    init_params_.depth_maximum_distance = 20;

    runtime_params_.measure3D_reference_frame = REFERENCE_FRAME::CAMERA;
    // Object Detection Parameters
    detection_params_.enable_tracking = true;
    detection_params_.enable_mask_output = false;
    detection_params_.detection_model = sl::DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

    cam_to_robot_.setIdentity();
    cam_to_robot_.tx = CAM_TO_ROBOT_X;
    cam_to_robot_.ty = CAM_TO_ROBOT_Y;
}

Zed::~Zed() {}

bool Zed::successful_grab() {
    return (zed_.grab(runtime_params_) == ERROR_CODE::SUCCESS);
}


bool Zed::open_camera() {
    auto return_state = zed_.open(init_params_);
    calibration_params_ = zed_.getCameraInformation().camera_configuration.calibration_parameters;
    return (return_state == ERROR_CODE::SUCCESS);
}

bool Zed::enable_tracking() {
    PositionalTrackingParameters tracking_params;
    if (!zed_.isOpened()) {
        error("Opening vision failed");
        return false;
    }
    tracking_params.enable_area_memory = true;
    sl::Transform initial_position;
    zed_.enablePositionalTracking(tracking_params);
    return true;
}

bool Zed::enable_tracking(Eigen::Vector3d init_pose) {
    PositionalTrackingParameters tracking_params;
    if (!zed_.isOpened()) {
        error("Opening vision failed");
        return false;
    }
    tracking_params.enable_area_memory = true;
    sl::Transform initial_position;
    initial_position.setTranslation(sl::Translation(init_pose(0), init_pose(1), 0));
    tracking_params.initial_world_transform = initial_position;
    zed_.enablePositionalTracking(tracking_params);
    return true;
}

bool Zed::enable_object_detection() {
    if (zed_.enableObjectDetection(detection_params_) != sl::ERROR_CODE::SUCCESS) {
        error("Object Detection Failed");
        return false;
    }
    return true;
}

void Zed::fetch_measurements() {
    if (successful_grab()) {
        zed_.retrieveImage(measurements_.left_image, VIEW::LEFT);
        measurements_.timestamp = measurements_.left_image.timestamp;
        zed_.retrieveMeasure(measurements_.depth_map, MEASURE::DEPTH);
        zed_.retrieveMeasure(measurements_.point_cloud, MEASURE::XYZRGBA);
        zed_.getPosition(measurements_.camera_pose);
    }
}


void Zed::input_custom_objects(std::vector<sl::CustomBoxObjectData> objects_in) {
    zed_.ingestCustomBoxObjects(objects_in);
}

sl::Mat Zed::get_depth_map() const {
    return measurements_.depth_map;
}

sl::Mat Zed::get_point_cloud() const {
    return measurements_.point_cloud;
}

sl::float3 Zed::get_position_from_pixel(int x, int y) {
    sl::float4 point3d;
    get_point_cloud().getValue(x, y, &point3d);
    return {point3d.x, point3d.y, point3d.z};
}

sl::float3 Zed::get_position_from_pixel(const cv::Point &p) {
    return get_position_from_pixel(p.x, p.y);
}

double Zed::get_distance_from_point(const sl::float3& p) {
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

sl::Transform Zed::get_calibration_stereo_transform() const {
    return calibration_params_.stereo_transform;
}

sl::Mat Zed::get_left_image() const {
    return measurements_.left_image;
}

sl::Pose Zed::get_camera_pose() const {
    return measurements_.camera_pose;
}

Timestamp Zed::get_measurement_timestamp() const {
    return measurements_.timestamp;
}

void Zed::close() {
    zed_.close();
}


