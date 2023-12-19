//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"
#include "inference/onnxruntime_inference_engine.hpp"
#include "utils/logger.hpp"
#include <chrono>

VisionContainer::VisionContainer() : 
    vision_runner_(nullptr),
    task_manager_(std::make_shared<TaskManager>()) {
}

void VisionContainer::init() {
    LOG_INFO("Init Vision container called");
    ONNXRuntimeInferenceEngine engine;
    LOG_INFO("Loading model");
    engine.load_model("yolov8n.onnx");
    LOG_INFO("Model loaded");
    cv::Mat mat = cv::imread("bus.jpg");
    engine.run_inference(mat);
}

void VisionContainer::run() {
    using namespace std::chrono_literals;
    init();
    vision_runner_ = task_manager_->create_task<VisionRunner>(0.001, "vision-runner");
    vision_runner_->start();
    for (;;) {
        std::this_thread::sleep_for(1s);
    }
}

VisionContainer::~VisionContainer() {
}
