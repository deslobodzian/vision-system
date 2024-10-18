#ifdef CUDA
#ifndef VISION_SYSTEM_COMMON_CUDA_OPS_H
#define VISION_SYSTEM_COMMON_CUDA_OPS_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device.h>
#include <logger.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr) {                                                               \
    cudaError_t error_code = callstr;                                                       \
    if (error_code != cudaSuccess) {                                                        \
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;  \
        assert(0);                                                                          \
    }                                                                                       \
}
#endif /* CUDA_CHECK */

class CudaAllocator : public Allocator {
public:
    CudaAllocator() = default;
    ~CudaAllocator() = default;
    void* alloc(size_t size) override {
        void* data;
        CUDA_CHECK(cudaMalloc(&data, size));
        return data;
    }

    void free(void* ptr) override {
        LOG_DEBUG("Freeing pointer: ", ptr);
        CUDA_CHECK(cudaFree(ptr));
    }

    void copy_in(void* dst, void* src, size_t size) override {
        LOG_ERROR("CUDA NOT IMPLEMENTED", src, dst, size);
    }

    void copy_out(void* src, void* dst, size_t size) override {
        LOG_ERROR("CUDA NOT IMPLEMENTED", src, dst, size);
    }
};

#endif /* VISION_SYSTEM_COMMON_CUDA_OPS_H */
#endif /* CUDA */
