#ifndef VISION_SYSTEM_CAMERA_HPP
#define VISION_SYSTEM_CAMERA_HPP

enum CAMERA_TYPE {
    MONOCULAR,
    ZED
};

class ICamera {
public:
    virtual int open_camera() = 0;
    virtual void close() = 0;
    virtual void fetch_measurements() = 0;
    virtual const CAMERA_TYPE get_camera_type() const = 0;
    virtual ~ICamera() = default;
};

#endif /* VISION_SYSTEM_CAMERA_HPP */
