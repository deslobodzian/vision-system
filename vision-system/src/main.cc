#include "logger.h"
#include "zed.h"
#include "system_container.h"
#include <cscore_oo.h>
#include <memory>
#include <networktables/NetworkTableInstance.h>
#include "cv_camera.h"
#include "zed_utils.h"

#include <cscore.h>
#include <cscore_cv.h>
#include <opencv2/imgproc.hpp>

int main() {
    logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);
    using namespace std::chrono;
    using namespace std::chrono_literals;
    LOG_INFO("Hello World!");

    nt::NetworkTableInstance instance_ = nt::NetworkTableInstance::GetDefault();
    instance_.GetTable("test")->PutBoolean("nt_test", false);
    LOG_INFO(&instance_);
    auto container = std::make_unique<SystemContainer>();
    // CVCamera cv_cam(0);

    // cv_cam.fetch();
    // cv::Mat cv_img = cv_cam.get_image();
    // int h = cv_img.rows;
    // int w = cv_img.cols;
    // LOG_DEBUG("Image data {", w, "x", h, "}");
    // std::this_thread::sleep_for(0.01s);



    // for (int i = 0; i < 10 / 0.01; i++) {
    //     cv_cam.fetch();
    //     cv_img = cv_cam.get_image();
    //     rgb_source.PutFrame(cv_img);

    //     h = cv_img.rows;
    //     w = cv_img.cols;
    //     LOG_DEBUG("Image data {", w, "x", h, "}");
    //     std::this_thread::sleep_for(0.01s);
    // }


    #ifdef CUDA
    LOG_INFO("CUDA enabled, initializing ZED camera...");
    //container->run();
    //container->list_current_tasks();
    auto cam = std::make_unique<ZedCamera>();
    sl::InitParameters params{};
    params.camera_resolution = sl::RESOLUTION::HD1080;
    params.coordinate_units = sl::UNIT::METER;
    params.depth_mode = sl::DEPTH_MODE::NEURAL;
    params.depth_stabilization = 1;

    int rgb_port = 1181;
    sl::Resolution res = sl::getResolution(params.camera_resolution);
    cs::CvSource rgb_source = cs::CvSource("ZED_RGB", cs::VideoMode::kMJPEG, res.height, res.width, 30);
    cs::MjpegServer rgb_server = cs::MjpegServer("RGB_Server", rgb_port);
    rgb_server.SetSource(rgb_source);
    LOG_INFO("RGB stream: source http://localhost:", rgb_port, "/?action=stream\n");

    int depth_port = 1191;
    cs::CvSource depth_source = cs::CvSource("ZED_Depth", cs::VideoMode::kMJPEG, res.height, res.width, 30);
    cs::MjpegServer depth_server = cs::MjpegServer("Depth_Server", depth_port);
    depth_server.SetSource(depth_source);
    LOG_INFO("Depth stream: source http://localhost:", depth_port, "/?action=stream\n");

    sl::PositionalTrackingParameters tracking_params{};
    tracking_params.enable_imu_fusion = true;
    tracking_params.enable_area_memory = true;
    sl::RuntimeParameters runtime_params = {};
    //runtime_params.confidence_threshold = 95;
    //runtime_params.texture_confidence_threshold = 95;
    cam->open(params, runtime_params);
    cam->enable_tracking(tracking_params);
    LOG_INFO(cam->camera_status_string());
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH);
    cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH | MeasurementType::SENSORS);
    cam->fetch_measurements(MeasurementType::IMAGE);

    //cam->enable_streaming();
    for (int i = 0; i < 10 / 0.01; i++) {
        cam->fetch_measurements(MeasurementType::IMAGE | MeasurementType::DEPTH | MeasurementType::POSE | MeasurementType::DEPTH_COLOR, sl::MEM::CPU);
        cv::Mat img = sl_to_cv(cam->get_measurements().left_image);
        rgb_source.PutFrame(img);
        cv::Mat depth = sl_to_cv(cam->get_measurements().depth_map);
        cv::Mat depth_normalized;
        double min_val, max_val;
        cv::minMaxLoc(depth, &min_val, &max_val);
        
        depth.convertTo(depth_normalized, CV_8UC1, -255.0 / (max_val - min_val), 255.0 * max_val / (max_val - min_val));
        
        cv::Mat depth_colored;
        cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);
        
        depth_source.PutFrame(depth_colored);
    }

    //LOG_INFO(cam->camera_status_string());
    //cam->disable_streaming();
    cam->close();
    #endif /* CUDA */

    return 0;
}
