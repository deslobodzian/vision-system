#pragma once
#include <string>
#include <memory>
#include "camera_spec.h"

enum class CameraType {
    GENERIC,
    ZED,
    WEBCAM,
    REALSENSE,
    ARGUS
};

class Camera {
public:
    virtual ~Camera() = default;

    virtual int open() = 0;
    virtual void close() = 0;
    virtual bool is_open() = 0;
    virtual std::string get_name() const = 0;
    virtual CameraType get_type() const = 0;

    // Get camera specifications
    virtual const camera::CameraSpec& get_specs() const = 0;

    // Dynamic cast helper
    template<typename T>
    bool is_type() const {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    // Factory method to create camera from ID
    static std::unique_ptr<Camera> create_camera(const std::string& camera_id);

    // Factory method to create camera from specs
    static std::unique_ptr<Camera> create_camera(const camera::CameraSpec& specs);
};
