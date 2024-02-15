#ifdef WITH_CUDA
#include "vision/zed.hpp"
#include "utils/logger.hpp"

ZedCamera::ZedCamera()
    : init_params_{}, runtime_params_{}, obj_detection_params_{},
    obj_rt_params_{}, batch_params_{}, calibration_params_{}, measurements_{},
    detected_objects_{} {}

    ZedCamera::ZedCamera(const std::string &svo_path) : ZedCamera() {
        svo_ = svo_path;
        LOG_INFO("Setting from svo file");
        init_params_.input.setFromSVOFile(svo_.c_str());
    }

void ZedCamera::configure(const zed_config &config) {
    if (config.serial_number != 0) {
        init_params_.input.setFromSerialNumber(config.serial_number);
    }
    init_params_.camera_resolution = config.res;
    init_params_.camera_fps = config.fps;
    init_params_.camera_image_flip = config.flip_camera;
    init_params_.depth_mode = config.depth_mode;
    init_params_.sdk_verbose = config.sdk_verbose;
    init_params_.coordinate_system = config.coordinate_system;
    init_params_.coordinate_units = config.units;
    init_params_.depth_maximum_distance = config.max_depth;

    runtime_params_.measure3D_reference_frame = config.reference_frame;

    obj_detection_params_.enable_tracking = config.enable_tracking;
    obj_detection_params_.enable_segmentation = config.enable_segmentation;
    obj_detection_params_.detection_model = config.model;
    obj_detection_params_.prediction_timeout_s = config.prediction_timeout_s;

    obj_rt_params_.detection_confidence_threshold =
        config.detection_confidence_threshold;
    batch_params_.enable = config.enable_batch;
    batch_params_.id_retention_time = config.id_retention_time;
    batch_params_.latency = config.batch_latency;

    obj_detection_params_.batch_parameters = batch_params_;

    memory_type_ = config.default_memory;
}

bool ZedCamera::successful_grab() {
    grab_state_ = zed_.grab();
    return (grab_state_ == sl::ERROR_CODE::SUCCESS);
}

int ZedCamera::open() {
    auto return_state = zed_.open(init_params_);
    if (return_state == sl::ERROR_CODE::SUCCESS) {
        return 0;
    } else {
        LOG_ERROR("Failed to open zed camera: ", return_state);
    }
    return -1;
}

int ZedCamera::enable_tracking() {
    sl::PositionalTrackingParameters tracking_params;
    if (!zed_.isOpened()) {
        LOG_ERROR("Camera is not open!");
        return -1;
    }
    tracking_params.enable_area_memory = true;
    sl::Transform initial_position;
    zed_.enablePositionalTracking(tracking_params);
    return 0;
}

int ZedCamera::enable_object_detection() {
    if (zed_.enableObjectDetection(obj_detection_params_) !=
            sl::ERROR_CODE::SUCCESS) {
        LOG_ERROR("Failed to enable object detection!");
        return -1;
    }
    return 0;
}

void ZedCamera::fetch_measurements(const MeasurementType &type) {
    LOG_DEBUG("Using memory type: ", memory_type_);
    if (successful_grab()) {
        switch (type) {
            case MeasurementType::ALL:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                zed_.getSensorsData(measurements_.sensor_data, sl::TIME_REFERENCE::IMAGE);
                measurements_.imu_data = measurements_.sensor_data.imu;
                measurements_.timestamp = measurements_.left_image.timestamp;
                zed_.retrieveMeasure(measurements_.depth_map, sl::MEASURE::DEPTH);
                zed_.getPosition(measurements_.camera_pose);
                zed_.retrieveMeasure(measurements_.point_cloud, sl::MEASURE::XYZ);
                break;
            case MeasurementType::IMAGE:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                break;
            case MeasurementType::DEPTH:
                zed_.retrieveMeasure(measurements_.depth_map, sl::MEASURE::DEPTH);
                break;
            case MeasurementType::OBJECTS:
                zed_.retrieveObjects(detected_objects_, obj_rt_params_);
                break;
            case MeasurementType::SENSORS:
                zed_.getSensorsData(measurements_.sensor_data, sl::TIME_REFERENCE::IMAGE);
                measurements_.imu_data = measurements_.sensor_data.imu;
                break;
            case MeasurementType::IMAGE_AND_SENSORS:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                zed_.getSensorsData(measurements_.sensor_data, sl::TIME_REFERENCE::IMAGE);
                measurements_.timestamp = measurements_.left_image.timestamp;
                break;
            case MeasurementType::IMAGE_AND_DEPTH:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                zed_.retrieveMeasure(measurements_.depth_map, sl::MEASURE::DEPTH);
                break;
            case MeasurementType::IMAGE_AND_POINT_CLOUD:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                zed_.retrieveMeasure(measurements_.point_cloud, sl::MEASURE::XYZ);
                break;
            case MeasurementType::IMAGE_AND_OBJECTS:
                zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
                zed_.retrieveObjects(detected_objects_, obj_rt_params_);
                break;
        }
    }
}

sl::Mat ZedCamera::get_depth_map() const { return measurements_.depth_map; }

const sl::Mat &ZedCamera::get_point_cloud() const { return measurements_.point_cloud; }

const sl::Mat &ZedCamera::get_left_image() const {
    return measurements_.left_image;
}

void ZedCamera::get_left_image(sl::Mat &img) const {
    img = measurements_.left_image;
}

sl::Pose ZedCamera::get_camera_pose() const {
    return measurements_.camera_pose;
}

sl::Timestamp ZedCamera::get_measurement_timestamp() const {
    return measurements_.timestamp;
}

sl::SensorsData::IMUData ZedCamera::get_imu_data() const {
    return measurements_.imu_data;
}

void ZedCamera::ingest_custom_objects(
        std::vector<sl::CustomBoxObjectData> &objs) {
    zed_.ingestCustomBoxObjects(objs);
}

const sl::Objects &ZedCamera::retrieve_objects() const { return detected_objects_; }

void ZedCamera::set_memory_type(const sl::MEM &memory) { memory_type_ = memory; }

const sl::Resolution ZedCamera::get_resolution() const {
    LOG_DEBUG("Init_params res: ", init_params_.camera_resolution);
    return sl::getResolution(init_params_.camera_resolution);
}

// requires camera to already be open
const sl::Resolution ZedCamera::get_svo_resolution() {
    if (successful_grab()) {
        zed_.retrieveImage(measurements_.left_image, sl::VIEW::LEFT, memory_type_);
    }
    sl::Resolution res = measurements_.left_image.getResolution();
    // reset frame position
    zed_.setSVOPosition(0);
    return res;
}

const sl::ERROR_CODE ZedCamera::get_grab_state() { return grab_state_; }

void ZedCamera::close() { zed_.close(); }

CUstream_st* ZedCamera::get_cuda_stream() {
    return zed_.getCUDAStream();
}

void ZedCamera::synchronize_cuda_stream() {
    cudaStreamSynchronize(zed_.getCUDAStream());
}

ZedCamera::~ZedCamera() { close(); };

#endif /* WITH_CUDA */
