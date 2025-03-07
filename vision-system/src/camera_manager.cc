#include "camera_manager.h"
#include "logger.h"

CameraManager::CameraManager() {
}

CameraManager::~CameraManager() {
    close_all();
}

bool CameraManager::detect_cameras() {

    // TODO: Use this later
    // bool found_webcams = detect_webcams();
    // bool found_zed = detect_zed_cameras();

    return !cameras_.empty();
}

bool CameraManager::detect_webcams() {
  // TODO DO
    bool found_any = false;
    return found_any;
}

bool CameraManager::detect_zed_cameras() {
    // TODO Zed camera finder
    #ifdef ZED_SDK_AVAILABLE
    #endif

    return false; // No ZED cameras found in this placeholder
}

bool CameraManager::add_camera(const std::string& id, std::unique_ptr<Camera> camera) {
    if (!camera) {
        LOG_ERROR("Cannot add null camera with ID: " + id);
        return false;
    }

    auto [it, inserted] = cameras_.emplace(id, std::move(camera));

    if (inserted) {
        LOG_DEBUG("Added camera with ID: " + id);
    } else {
        LOG_ERROR("Camera with ID already exists: " + id);
    }

    return inserted;
}

bool CameraManager::remove_camera(const std::string& id) {
    auto it = cameras_.find(id);
    if (it == cameras_.end()) {
        LOG_INFO("Cannot remove non-existent camera with ID: " + id);
        return false;
    }

    it->second->close();
    cameras_.erase(it);
    LOG_DEBUG("Removed camera with ID: " + id);
    return true;
}

Camera* CameraManager::get_camera(const std::string& id) {
    auto it = cameras_.find(id);
    return it != cameras_.end() ? it->second.get() : nullptr;
}

std::vector<std::string> CameraManager::list_camera_ids() const {
    std::vector<std::string> ids;
    ids.reserve(cameras_.size());

    for (const auto& [id, _] : cameras_) {
        ids.push_back(id);
    }

    return ids;
}

std::vector<std::pair<std::string, CameraType>> CameraManager::list_cameras_with_types() const {
    std::vector<std::pair<std::string, CameraType>> result;
    result.reserve(cameras_.size());

    for (const auto& [id, camera_ptr] : cameras_) {
        result.emplace_back(id, camera_ptr->get_type());
    }

    return result;
}

std::vector<std::pair<std::string, camera::CameraSpec>> CameraManager::list_cameras_with_specs() const {
    std::vector<std::pair<std::string, camera::CameraSpec>> result;
    result.reserve(cameras_.size());

    for (const auto& [id, camera_ptr] : cameras_) {
        result.emplace_back(id, camera_ptr->get_specs());
    }

    return result;
}

void CameraManager::close_all() {
    for (auto& [id, camera_ptr] : cameras_) {
        if (camera_ptr->is_open()) {
            LOG_DEBUG("Closing camera: " + id);
            camera_ptr->close();
        }
    }
}

const camera::CameraSpec* CameraManager::get_camera_specs(const std::string& id) const {
    auto it = cameras_.find(id);
    if (it != cameras_.end()) {
        return &(it->second->get_specs());
    }
    return nullptr;
}

std::string CameraManager::generate_camera_id(const camera::CameraSpec& specs) const {
    std::string backend = camera::backend_to_string(specs.backend_);
    std::string id_base = backend + "_" + std::to_string(specs.id_);

    std::string id = id_base;
    int counter = 1;

    while (cameras_.find(id) != cameras_.end()) {
        id = id_base + "_" + std::to_string(counter++);
    }

    return id;
}
