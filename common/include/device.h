#ifndef VISION_SYSTEM_COMMON_DEVICE_H
#define VISION_SYSTEM_COMMON_DEVICE_H

#include <cassert>
#include <cstddef>
#include <typeinfo>
#include <vector>
#include <logger.h>

enum class SupportedDevice {
    NV_CUDA,
    CPU,
};

const static std::vector<SupportedDevice> supported_devices {
    SupportedDevice::NV_CUDA, SupportedDevice::CPU
};

class Device {
public:
    static std::vector<SupportedDevice> get_available_devices() {
        std::vector<SupportedDevice> devices;
        for (SupportedDevice device : supported_devices) {
            switch (device) {
                case SupportedDevice::NV_CUDA:
                    #ifdef CUDA
                    LOG_DEBUG("Added CUDA to available devices");
                    devices.push_back(device);
                    #endif /* CUDA */
                    continue;
                case SupportedDevice::CPU:
                    devices.push_back(device);
                    continue;
                default:
                    assert("Using default something is wrong as you have a cpu?");
            }
        }
        return devices;
    }
};

template <typename T>
class Buffer {
public:
    Buffer(const Device& device, size_t count) :
        device_(device), buffer_size_(sizeof(T) * count) {
        LOG_DEBUG("Generating Buffer on device: ", device, " with datatype: ", typeid(T).name(), "with count: ", count);
    };
    Buffer* allocate();
    size_t buffer_size() const { return buffer_size_; };
private:
    Device device_;
    size_t buffer_size_;
};

/*
 * Allocator will handle all memory allocation for different devices
 */
class Allocator {
public:
    virtual void* alloc(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void copy_in(void* dst, void* src, size_t size) = 0;
    virtual void copy_out(void* src, void* dst, size_t size) = 0;
};


// class MallocAllocator : Allocator {
//     std::unique_ptr<void> alloc(size_t size) {
//
//     }
// };
#endif /* VISION_SYSTEM_COMMON_DEVICE_H */
