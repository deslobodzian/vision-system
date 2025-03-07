#include "camera_manager.h"
#include <algorithm>

CameraManager::CameraManager() {
}

CameraManager::~CameraManager() {
    close_all();
}

bool CameraManager::detect_cameras() {
    cameras_.clear();
    return !cameras_.empty();
}

bool CameraManager::add_camera(const std::string& id, std::unique_ptr<Camera> camera) {
    if (!camera) {
        return false;
    } 

    auto [it, inserted] = cameras_.emplace(id, std::move(camera));
    return inserted;
}

bool CameraManager::remove_camera(const std::string& id) {
    auto it = cameras_.find(id);
    if (it == cameras_.end()) {
        return false;
    }

    it->second->close();
    cameras_.erase(it);
    return true;
}

Camera* CameraManager::get_camera(const std::string& id) {
    auto it = cameras_.find(id);
    return it != cameras_.end() ? it->second.get() : nullptr;
}

void CameraManager::close_all() {
    for (auto& [id, camera] : cameras_) {
        camera->close();
    }
}


