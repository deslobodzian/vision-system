#include <logger.h>
#include <gtest/gtest.h>
#include "device_manager.h"
#include <cuda_runtime.h>

TEST(CPUDeviceTest, DeviceTests) {
   auto dev = DeviceManager::instance().get_device("CPU");
   auto dev_same = DeviceManager::instance().get_device("CPU");
   
   #if defined(CUDA_ENABLED)
       int deviceCount;
       cudaGetDeviceCount(&deviceCount);
       LOG_DEBUG("Found ", deviceCount, " CUDA device(s)");
       
       for (int i = 0; i < deviceCount; i++) {
           cudaDeviceProp props;
           cudaGetDeviceProperties(&props, i);
           LOG_DEBUG("Device ", i, ":", props.name);
           LOG_DEBUG("Compute Capability ", props.major, props.minor);
           LOG_DEBUG("Memory (MB) ", props.totalGlobalMem / (1024 * 1024));
       }
   #endif

   EXPECT_TRUE(dev == dev_same);
}