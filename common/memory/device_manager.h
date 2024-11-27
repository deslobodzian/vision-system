#ifndef VISION_SYSTEM_COMMON_DEVICE_MANAGER_H
#define VISION_SYSTEM_COMMON_DEVICE_MANAGER_H

#include <string>
#include <map>
#include <memory>

#include "device.h"
#include "cpu_device.h"
#include "logger.h"

/**
 * Handles global creation of devices
 **/

// namespace common::device_tracker {

class DeviceManager {
 public:
  static DeviceManager& instance() {
    static DeviceManager instance;
    return instance;
  }
  
  std::shared_ptr<Device> get_device(const std::string& device) {
    if (devices_[device] == nullptr) {
      bool ret = create_device(device);
      LOG_DEBUG("Created ", device, " returned ", ret);
    } else {
      LOG_DEBUG("Device ", device, " already created");
    } 
    return devices_[device];
  }

 private:
  std::map<std::string, std::shared_ptr<Device>> devices_ = {
    {"CPU", nullptr},
    {"CUDA", nullptr},
  };

  bool create_device(const std::string& dev) {
    if (devices_.count(dev) == 0) {
      LOG_ERROR("Device {", dev, "} is not supported");
      return false;
    }
    if (devices_[dev] != nullptr) {
      LOG_ERROR("Device {", dev, "} is already created");
      return false;
    }
    if (dev == "CPU") {
      devices_[dev] = std::make_shared<CPUDevice>();
    } else if (dev == "CUDA") {
      LOG_ERROR("NOT IMPLEMENTED");
    }
    return true;
  }
};

// } /* namespace common::device_tracker */
#endif /* VISION_SYSTEM_COMMON_DEVICE_MANAGER_H */
