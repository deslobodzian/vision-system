#ifndef VISION_SYSTEM_COMMON_DEVICE_MANAGER_H
#define VISION_SYSTEM_COMMON_DEVICE_MANAGER_H

#include <string>
#include <map>
#include <memory>

#include "device.h"
#include "logger.h"

#include "cpu_device.h"

#ifdef CUDA
#include "cuda_device.h"
#endif /* CUDA */

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
    auto iterator = devices_.find(device);
    if (iterator == devices_.end()) {
      std::string err_str = "Device {" + device + "} is not supported";
      LOG_ERROR(err_str);
      throw std::runtime_error(err_str);
    }

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
#ifdef CUDA
    {"CUDA", nullptr},
#endif /* CUDA */
  };

  bool create_device(const std::string& dev) {
    auto iterator = devices_.find(dev);
    if (iterator == devices_.end()) {
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
