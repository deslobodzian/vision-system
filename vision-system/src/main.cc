#include "logger.h"
#include "zed.h"
#include "system_container.h"

#include <memory>


int main() {
    logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);
    using namespace std::chrono;
    using namespace std::chrono_literals;
    LOG_INFO("Hello World!");
    auto container = std::make_unique<SystemContainer>();
    //container->run();
    //container->list_current_tasks();
    auto cam = std::make_unique<ZedCamera>();
    sl::InitParameters params{};
    params.camera_resolution = sl::RESOLUTION::HD1080;
    cam->open(params);
    LOG_INFO(cam->camera_status_string());
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH);
    LOG_INFO("\n");
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH | MeasurementType::SENSORS);
    LOG_INFO("\n");
    cam->fetch_measurements(MeasurementType::IMAGE);

    for (int i=0; i<10; i++) {
        LOG_INFO("SLEEPING");
        std::this_thread::sleep_for(1s);
    }

    LOG_INFO(cam->camera_status_string());
    return 0;
}