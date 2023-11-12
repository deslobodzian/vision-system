//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"
#include "inference/onnxruntime_inference_engine.hpp"
#include <chrono>

VisionContainer::VisionContainer() : 
    vision_runner_(nullptr),
    task_manager_(std::make_shared<TaskManager>()) {
}

void VisionContainer::init() {
    printf("Init Vision container called\n");
    ONNXRuntimeInferenceEngine engine;
    printf("Loading model\n");
    engine.load_model("yolov8n.onnx");
    printf("Model loaded\n");
}

void VisionContainer::run() {
    using namespace std::chrono_literals;
    init();
    vision_runner_ = task_manager_->create_task<VisionRunner>(0.001, "vision-runner");
    vision_runner_->start();
    for (;;) {
        std::this_thread::sleep_for(10s);
    }
}

VisionContainer::~VisionContainer() {
}
