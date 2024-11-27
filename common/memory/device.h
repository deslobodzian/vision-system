#ifndef VISION_SYSTEM_COMMON_DEVICE_H
#define VISION_SYSTEM_COMMON_DEVICE_H

#include "allocator.h"

class Device {
 public:
  Device() = default;
  Device(const Device&) = default;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = default;
  Device& operator=(Device&&) = delete;
  virtual ~Device() = default;
  virtual Allocator* allocator() = 0;
};

#endif /* VISION_SYSTEM_COMMON_DEVICE_H */
