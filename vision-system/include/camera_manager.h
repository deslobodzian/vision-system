#pragma once

#include <map>
#include <vector>
#include <string>
#include "camera.h"

/* Manager to handle camemeras detected */

class CameraManager {
public:
    CameraManager();
    ~CameraManager();

    bool detect_cameras();
    bool add_camera(const std::string& id, std::unique_ptr<Camera> camera);
    bool remove_camera(const std::string& id);

    Camera* get_camera(const std::string& id);

    template<typename T>
    T* get_camera_as(const std::string& id) {
        static_assert(std::is_base_of<Camera, T>::value, "T must be derived from Camera");

        auto it = cameras_.find(id);
        if (it == cameras_.end()) {
            return nullptr;
        }

        if (!it->second->is_type<T>()) {
            return nullptr;
        }

        return dynamic_cast<T*>(it->second.get());
    }

    template<typename T>
    std::vector<std::string> find_cameras_of_type() {
        static_assert(std::is_base_of<Camera, T>::value, "T must be derived from Camera");
        std::vector<std::string> res;

        for (const auto& [id, camera] : camera_) {
            if (camera->is_type<T>()) {
                res.push_back(id);
            }
        }

        return res;
    }

    void close_all();

    template<typename T> 
    bool is_camera_type(const std::string& id) {
        static_assert(std::is_base_of<Camera, T>::value, "T must be derived from Camera");
        Camera* camera = get_camera(id);
        return camera && camera->is_type<T>();
    }

    std::vector<std::string> list_camera_ids() const; // Maybe want this to be dynaimc
    std::vector<std::pair<std::string, CameraType>> list_cameras_with_types() const;
    
private:
    std::map<std::string, std::unique_ptr<Camera>> cameras_;
};