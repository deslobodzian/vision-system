#ifdef CUDA
#ifndef VISION_SYSTEM_COMMON_CUDA_OPS_H
#define VISION_SYSTEM_COMMON_CUDA_OPS_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device.h>
#include <logger.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr) {                                                                                   \
    cudaError_t error_code = callstr;                                                                           \
    if (error_code != cudaSuccess) {                                                                            \
        std::cerr << "CUDA error " << cudaGetErrorString(error_code) << " at " << __FILE__ << ":" << __LINE__;  \
        assert(0);                                                                                              \
    }                                                                                                           \
}
#endif /* CUDA_CHECK */

class CudaAllocator : public Allocator {
public:
    CudaAllocator() = default;
    ~CudaAllocator() = default;
    void* alloc(size_t size) override {
        void* data;
        LOG_DEBUG("Alloc: ", data, " with size: ", size);
        cudaError_t cudaStatus = cudaMalloc(&data, size);
        if (cudaStatus != cudaSuccess) {
            LOG_ERROR("Failed to alloc buffer");
        }
        LOG_DEBUG("Allocated");
        return data;
    }

    void free(void* ptr) override {
        LOG_DEBUG("Freeing pointer: ", ptr);
        CUDA_CHECK(cudaFree(ptr));
    }

    void copy_in(void* dst, void* src, size_t size) override {
        LOG_DEBUG("dst ptr: ", dst);
        LOG_DEBUG("src ptr: ", src);
        LOG_DEBUG("size: ", size);

        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }

    void copy_out(void* src, void* dst, size_t size) override {
        LOG_DEBUG("dst ptr: ", dst);
        LOG_DEBUG("src ptr: ", src);
        LOG_DEBUG("size: ", size);

        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
};

class CudaDevice : public Device {
public:
    CudaDevice() : Device(new CudaAllocator) {
        LOG_DEBUG("Created CUDA Device");
        CUDA_CHECK(cudaSetDevice(0));
    }

    DeviceType type() const override { return DeviceType::NV_CUDA; }
    std::string name() const override { return "CUDA"; }
};

#endif /* VISION_SYSTEM_COMMON_CUDA_OPS_H */
#endif /* CUDA */
