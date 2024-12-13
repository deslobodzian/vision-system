#ifndef VISION_SYSTEM_COMMON_CPU_DEVICE_H
#define VISION_SYSTEM_COMMON_CPU_DEVICE_H
#include "device.h"

class MallocAllocator : public Allocator {
    void* alloc(size_t size) override {
        void* data = ::operator new(size);
        return data;
    }

    void free(void* ptr) override {
        LOG_DEBUG("Freeing pointer: ", ptr);
        ::operator delete(ptr);
    }

    void copy_in(void* dst, void* src, size_t size) override {
        std::memcpy(dst, src, size);
    }

    void copy_out(void* src, void* dst, size_t size) override {
        std::memcpy(src, dst, size);
    }
 };

class CPUDevice : public Device {
public:
    CPUDevice() : Device(new MallocAllocator) {
        LOG_DEBUG("Created CPU Device");
    }
    DeviceType type() const override { return DeviceType::CPU; }
    std::string name() const override { return "CPU"; }
};
#endif /* VISION_SYSTEM_COMMON_CPU_DEVICE_H */
