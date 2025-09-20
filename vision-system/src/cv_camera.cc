#include "cv_camera.h"
#include "logger.h"

CVCamera::CVCamera(const int& id) : id_(id), camera_(id, cv::CAP_ANY) {
    LOG_DEBUG("Initialize camera id: ", id);
    if (!camera_.isOpened()) {
        LOG_ERROR("Failed to open camera: ", id, "!");
    }

}

bool CVCamera::fetch() {
    return camera_.read(image_mat_);
}
