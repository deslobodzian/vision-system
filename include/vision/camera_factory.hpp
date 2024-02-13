#ifndef VISION_SYSTEM_CAMERA_FACTORY
#define VISION_SYSTEM_CAMERA_FACTORY

#include "i_camera.hpp"
#include "zed.hpp"
#include "monocular_camera.hpp"

class CameraFactory {
public:
    static std::unique_ptr<ICamera>create_camera(const CAMERA_TYPE& type) {
        if (type == ZED) {
            return std::make_unique<ZedCamera>();
        }
        return std::make_unique<MonocularCamera>();
    }
private:
};

#endif /* VISION_SYSTEM_CAMERA_FACTORY */
