#include "logger.h"
//#include "zed.h"
#include "system_container.h"
#ifdef CUDA
#include "zed_publisher.h"
#endif /* CUDA */
#include <memory>
#include <networktables/NetworkTableInstance.h>

int main() {
    logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);
    using namespace std::chrono;
    using namespace std::chrono_literals;
    LOG_INFO("Hello World!");

    nt::NetworkTableInstance instance_ = nt::NetworkTableInstance::GetDefault();
    instance_.GetTable("test")->PutBoolean("nt_test", false);
    LOG_INFO(&instance_);
    auto container = std::make_unique<SystemContainer>();

    #ifdef CUDA
    LOG_INFO("CUDA enabled, initializing ZED camera...");
    ZedPublisher pub("tcp://*:5555");
    //container->run();
    //container->list_current_tasks();
    auto cam = std::make_unique<ZedCamera>();
    sl::InitParameters params{};
    params.camera_resolution = sl::RESOLUTION::HD1080;
    params.coordinate_units = sl::UNIT::METER;
    params.depth_mode = sl::DEPTH_MODE::ULTRA;
    params.depth_stabilization = 1;
    sl::PositionalTrackingParameters tracking_params{};
    tracking_params.enable_imu_fusion = true;
    tracking_params.enable_area_memory = true;
    sl::RuntimeParameters runtime_params = {};
    runtime_params.confidence_threshold = 50;
    runtime_params.texture_confidence_threshold = 50;
    cam->open(params, runtime_params);
    cam->enable_tracking(tracking_params);
    LOG_INFO(cam->camera_status_string());
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH);
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH | MeasurementType::SENSORS);
    cam->fetch_measurements(MeasurementType::IMAGE);
    //cam->enable_streaming();
    for (int i = 0; i < 10 / 0.01; i++) {
        cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH | MeasurementType::POSE, sl::MEM::CPU);
        LOG_DEBUG("Writing Measurements");
        pub.write_measurements(cam->get_measurements());
        std::this_thread::sleep_for(0.01s);
    }
    //LOG_INFO(cam->camera_status_string());
    //cam->disable_streaming();
    cam->close();
    #endif /* CUDA */

    return 0;
}
