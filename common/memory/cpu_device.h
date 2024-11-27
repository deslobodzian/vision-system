#ifndef VISION_SYSTEM_COMMON_CPU_DEVICE_H
#define VISION_SYSTEM_COMMON_CPU_DEVICE_H

#include "common/memory/device.h"
#include "common/memory/malloc_allocator.h"

class CPUDevice : public virtual Device {
 public:
  CPUDevice() : allocator_(new MallocAllocator()) {} 
  ~CPUDevice() override {
    delete allocator_;
  }

  Allocator* allocator() override { return allocator_; }

 private:
  MallocAllocator* allocator_;
};

#endif /* VISION_SYSTEM_COMMON_CPU_DEVICE_H */
