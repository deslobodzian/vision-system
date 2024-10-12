#ifndef VISION_SYSTEM_COMMON_DEVICE_H
#define VISION_SYSTEM_COMMON_DEVICE_H

/*
 * We will use the teminolagy that graphics programs use where a "device" is the GPU only
 */

#include <cstddef>
#include <memory>
#include <optional>

class Device {
private:
};

struct BufferConfig {
    bool cpu_access = false;
    std::optional<std::unique_ptr<void>> extenal_ptr = std::nullopt;
};

template <typename T>
class Buffer {
public:
    Buffer(const Device& device);
};

/*
 * Allocator will handle all memory allocation for different devices
 */
class Allocator {
public:
    virtual std::unique_ptr<void> alloc(size_t size) = 0;
    virtual void free() = 0;
    virtual void copy_in(void* dst, void* src, size_t size) = 0;
    virtual void copy_out(void* src, void* dst, size_t size) = 0;
};


// class MallocAllocator : Allocator {
//     std::unique_ptr<void> alloc(size_t size) {
//
//     }
// };
#endif /* VISION_SYSTEM_COMMON_DEVICE_H */
