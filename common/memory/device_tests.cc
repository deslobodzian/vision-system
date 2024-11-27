#include <logger.h>
#include <gtest/gtest.h>
#include "device_manager.h"

TEST(CPUDeviceTest, DeviceTests) {
    auto dev = DeviceManager::instance().get_device("CPU");
    LOG_INFO("Device ptr: ", dev);
    LOG_INFO("Device alloc ptr: ", dev->allocator());
    auto dev_test = DeviceManager::instance().get_device("CPU");
    LOG_INFO("Device test ptr: ", dev_test);
    LOG_INFO("Device test alloc ptr: ", dev_test->allocator());
    EXPECT_TRUE(dev == dev_test);
}
