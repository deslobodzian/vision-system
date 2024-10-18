#include <device.h>
#include <logger.h>
#include <algorithm>

#include <gtest/gtest.h>

TEST(DevicesTest, Devices) {
    std::vector<SupportedDevice> devices = Device::get_available_devices();
    #ifdef CUDA
        EXPECT_TRUE(std::find(devices.begin(), devices.end(), SupportedDevice::NV_CUDA) != devices.end());
    #endif /* CUDA */

    // This has to always be true
    EXPECT_TRUE(std::find(devices.begin(), devices.end(), SupportedDevice::CPU) != devices.end());
}

TEST(CudaBufferTest, BufferTests) {
}
