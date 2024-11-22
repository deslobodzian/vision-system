#ifndef VISION_SYSTEM_COMMON_DEVICE_TRACKER_H
#define VISION_SYSTEM_COMMON_DEVICE_TRACKER_H

#include <vector>
#include <string>
#include "device.h"

/**
 * Handles global creation of devices
**/


namespace common::device_tracker {
static const std::vector<std::string> supported_devices = {"CPU"};

class DeviceTracker {
 public:
  static DeviceTracker& instance() {
        static DeviceTracker instance;
        return instance;
  }
 private:
  std::vector<std::string> created_devices_;


};

//constexpr Device* create_device(const std::string& dev) {
//}
}  /* namespace common::device_tracker */
#endif /* VISION_SYSTEM_COMMON_DEVICE_TRACKER_H */
