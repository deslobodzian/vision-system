//
// Created by deslobodzian on 11/25/22.
//

#ifndef VISION_SYSTEM_CAMERA_HPP
#define VISION_SYSTEM_CAMERA_HPP

#include <opencv2/opencv.hpp>

enum CAMERA_TYPE {
    MONOCULAR,
    ZED
};
class GenericCamera {
public:
    virtual int open_camera(){return -1;};
    virtual void fetch_measurements(){};
    virtual CAMERA_TYPE get_camera_type() const = 0;

    virtual ~GenericCamera() = default;
private:
};

#endif //VISION_SYSTEM_CAMERA_HPP
