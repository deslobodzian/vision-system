//
// Created by ubuntuvm on 11/16/22.
//

#ifndef VISION_SYSTEM_NETWORK_TABLES_MANAGER_HPP
#define VISION_SYSTEM_NETWORK_TABLES_MANAGER_HPP

#include <networktables/NetworkTableInstance.h>
#include "nt_publisher.hpp"

class NTManager {
public:
    NTManager() {
        instance_ = nt::NetworkTableInstance::GetDefault();

        zed_publisher_ = NTPublisher(instance_, "Zed");
    };
    ~NTManager();

private:
    nt::NetworkTableInstance instance_;
    NTPublisher zed_publisher_;

};

#endif //VISION_SYSTEM_NETWORK_TABLES_MANAGER_HPP
