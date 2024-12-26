#ifndef VISION_SYSTEM_VISION_SYSTEM_ZED_H
#define VISION_SYSTEM_VISION_SYSTEM_ZED_H

#include <sl/Camera.hpp>
#include <sstream>

using namespace sl;

/* Use for caching measurements */
typedef struct {
    Timestamp timestamp;
    Mat left_image;
    Pose camera_pose;
    Mat depth_map;
    Mat point_cloud;
    SensorsData sensors_data;
} ZedMeasurements;

// Bit identification for types 
enum class MeasurementType {
    NONE = 0,
    IMAGE = 1 << 0,
    DEPTH = 1 << 1,
    SENSORS = 1 << 2,
    OBJECTS = 1 << 3,
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
    std::string camera_status_string();
    int open(const InitParameters& init_params);
    bool sucessfull_grab(); 
    int fetch_measurements(const MeasurementType& types, const sl::MEM& memory_type = sl::MEM::CPU);

private:
    ERROR_CODE grab_state_;

    Camera zed_;
    ZedMeasurements measurements_;
    InitParameters init_params_;
};
#endif /* VISION_SYSTEM_VISION_SYSTEM_ZED_H */