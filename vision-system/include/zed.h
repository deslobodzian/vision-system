#ifdef CUDA
#pragma once

#include <sl/Camera.hpp>
#include <sstream>

/* Use for caching measurements */
typedef struct {
    sl::Timestamp timestamp;
    sl::Mat left_image;
    sl::Pose camera_pose;
    sl::Mat depth_map;
    sl::Mat depth_color;
    sl::Mat point_cloud;
    sl::SensorsData sensors_data;
} ZedMeasurements;

// Bit identification for types
enum class MeasurementType {
    NONE = 0,
    IMAGE = 1 << 0,
    DEPTH = 1 << 1,
    SENSORS = 1 << 2,
    OBJECTS = 1 << 3,
    POSE = 1 << 4, 
    DEPTH_COLOR = 1 << 5,
};

inline MeasurementType operator|(MeasurementType a, MeasurementType b) {
    return static_cast<MeasurementType>(static_cast<int>(a) | static_cast<int>(b));
}

inline MeasurementType operator&(MeasurementType a, MeasurementType b) {
    return static_cast<MeasurementType>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool has_measurement(MeasurementType flags, MeasurementType check) {
    return (static_cast<int>(flags) & static_cast<int>(check)) == static_cast<int>(check);
}

class ZedCamera {
public:
    ZedCamera();
    ~ZedCamera();

    int open();
    void close();
    bool is_open();
    std::string get_name() const { return name_; }

    std::string camera_status_string();
    int open(const sl::InitParameters& init_params, const sl::RuntimeParameters& runtime_params);
    int enable_tracking(const sl::PositionalTrackingParameters& tracking_params);
    bool successful_grab(); 
    int fetch_measurements(const MeasurementType& types, const sl::MEM& memory_type = sl::MEM::CPU);
    int enable_streaming();
    void disable_streaming();
    const ZedMeasurements& get_measurements();

private:
    sl::ERROR_CODE grab_state_;

    std::string name_;
    sl::Camera zed_;
    ZedMeasurements measurements_;
    sl::InitParameters init_params_;
    sl::RuntimeParameters runtime_params_;

    bool tracking_enabled_;
};

#endif /* CUDA */
