#include <device.h>
#include <gtest/gtest.h>
#include <logger.h>
#include <cuda_ops.h>

#ifdef CUDA
TEST(AllocatorTests, CUDAAllocator) {
    CudaAllocator allocator;
    void* test = allocator.alloc(10);
}
#endif /* CUDA */

