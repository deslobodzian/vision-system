#pragma once

#include <string>
#include <memory>

enum class CameraType {
    GENERIC,
    ZED,
    WEBCAM
};

class Camera {
public:
    virtual ~Camera() {};

    virtual int open() = 0;
    virtual void close() = 0;
    virtual bool is_open() = 0;
    virtual std::string get_name() const = 0;

    virtual CameraType get_type() const = 0;

    template<typename T>
    bool is_type() const {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    static std::unique_ptr<Camera> create_camera(const std::string& camera_id);
};