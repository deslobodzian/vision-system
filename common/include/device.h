#ifndef VISION_SYSTEM_COMMON_DEVICE_H
#define VISION_SYSTEM_COMMON_DEVICE_H

#include <cassert>
#include <cstddef>
#include <cstring>
#include <typeinfo>
#include <vector>
#include <logger.h>

enum class DeviceType {
    NV_CUDA,
    CPU,
};

const static std::vector<DeviceType> supported_devices {
    DeviceType::NV_CUDA, DeviceType::CPU
};

const static std::vector<DeviceType> get_available_devices() {
    std::vector<DeviceType> devices;
    for (DeviceType device : supported_devices) {
        switch (device) {
            case DeviceType::NV_CUDA:
#ifdef CUDA
            LOG_DEBUG("Added CUDA to available devices");
            devices.push_back(device);
#endif /* CUDA */
            continue;
            case DeviceType::CPU:
                devices.push_back(device);
                continue;
            default:
                assert("Using default something is wrong as you have a cpu?");
        }
    }
    return devices;
}

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

class Device {
public:
    explicit Device(Allocator* allocator) : allocator_(std::move(allocator)) {
    }
    virtual ~Device() = default;

    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;


    Allocator* get_allocator() const { return allocator_; }
private:
    Allocator* allocator_;
};

class CPUDevice : public Device {
public:
    CPUDevice() : Device(new MallocAllocator) {
        LOG_DEBUG("Created CPU Device");
    }
    DeviceType type() const override { return DeviceType::CPU; }
    std::string name() const override { return "CPU"; }
};

template <typename T>
class Buffer {
public:
    Buffer(const Device& device, size_t count) :
        device_(device), count_(count), buffer_size_(sizeof(T) * count), data_(nullptr) {
        LOG_DEBUG("Generating Buffer on device: ", device.name(), " with datatype: ", typeid(T).name(), " with count: ", count);
    };
    ~Buffer() {
        if (data_) {
            LOG_DEBUG("Freeing ptr");
            device_.get_allocator()->free(data_);
        }
    }
    void allocate() {
        LOG_DEBUG("Allocating buffer");
        if (data_) {
            LOG_ERROR("Buffer already allocated!");
            throw std::runtime_error("Buffer already allocated!");
        }
        data_ = device_.get_allocator()->alloc(buffer_size_);
        LOG_DEBUG("Buffer allocated");
    }

    void copy_to_device(const T* host_data) {
        if (!data_) {
            LOG_ERROR("Buffer not allocated");
            throw std::runtime_error("Buffer not allocated");
        }
        device_.get_allocator()->copy_in(
            data_,
            const_cast<void*>(static_cast<const void*>(host_data)),
            buffer_size_
        );
    }

    void copy_to_host(T* host_data) {
        if (!data_) {
            LOG_ERROR("Buffer not allocated");
            throw std::runtime_error("Buffer not allocated");
        }
        device_.get_allocator()->copy_out(data_, host_data, buffer_size_);
    }

    size_t count() const { return count_; }
    size_t buffer_size() const { return buffer_size_; };
    void* ptr() const { return data_; }
    // This is more for CPU access
    T* data() const { return static_cast<T*>(data_); }

private:
    const Device& device_;
    size_t count_;
    size_t buffer_size_;
    void* data_;
};
#endif /* VISION_SYSTEM_COMMON_DEVICE_H */
