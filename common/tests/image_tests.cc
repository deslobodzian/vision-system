#include <image.h>
#include <logger.h>
#include <device_manager.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("CpuImageTest", "[image]") {
    auto cpu_dev = DeviceManager::instance().get_device("CPU");
    LOG_DEBUG("Creating image on device: ", cpu_dev->name());
    Image<double> image(*cpu_dev, 32, 64, 3);

    REQUIRE(image.height() == 32);
    REQUIRE(image.width() == 64);
    REQUIRE(image.channels() == 3);
    REQUIRE(image.stride() == 64 * 3 * sizeof(double));
    REQUIRE(image.size() == 32 * 64 * 3 * sizeof(double));
}

#ifdef CUDA
TEST_CASE("CUDAImageTest", "[image]") {
    auto dev= DeviceManager::instance().get_device("CUDA");
    LOG_DEBUG("Creating image on device: ", dev->name());
    Image<double> image(*dev, 32, 64, 3);

    REQUIRE(image.height() == 32);
    REQUIRE(image.width() == 64);
    REQUIRE(image.channels() == 3);
    REQUIRE(image.stride() == 64 * 3 * sizeof(double));
    REQUIRE(image.size() == 32 * 64 * 3 * sizeof(double));
}
#endif /* CUDA */
