#include <device.h>
#include <logger.h>
#include <algorithm>

#include <gtest/gtest.h>

TEST(DevicesTest, Devices) {
    std::vector<DeviceType> devices = get_available_devices();
    #ifdef CUDA
        EXPECT_TRUE(std::find(devices.begin(), devices.end(), SupportedDevice::NV_CUDA) != devices.end());
    #endif /* CUDA */

    // This has to always be true
    EXPECT_TRUE(std::find(devices.begin(), devices.end(), DeviceType::CPU) != devices.end());
}

TEST(CPUBufferTest, BufferTests) {

    auto cpu_device = new CPUDevice();
    LOG_INFO(cpu_device->name());

    Buffer<int> buff_int = Buffer<int>(*cpu_device, 10);
    Buffer<float> buff_float = Buffer<float>(*cpu_device, 10);

    buff_int.allocate();
    buff_float.allocate();

    int int_test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float float_test[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    buff_float.copy_to_host(float_test);
    buff_int.copy_to_host(int_test);

    EXPECT_TRUE(std::equal(buff_int.data(), buff_int.data() + buff_int.count(), int_test));
    EXPECT_TRUE(std::equal(buff_float.data(), buff_float.data() + buff_float.count(), float_test));
}

TEST(CudaBufferTest, BufferTests) {
}
