//
// Created by deslobodzian on 11/22/22.
//

#include "vision_container.hpp"
#include "inference/inference_engine_factory.hpp"
#include "utils/logger.hpp"
#include "inference/yolo.hpp"
#include <chrono>

VisionContainer::VisionContainer() : 
    vision_runner_(nullptr),
    task_manager_(std::make_shared<TaskManager>()) {
}
void drawBoundingBoxes(cv::Mat& image, const std::vector<BBoxInfo>& bboxes) {
    for (const auto& bbox : bboxes) {
        // Draw rectangle around detected object
        cv::rectangle(image, cv::Point(bbox.box.x1, bbox.box.y1), cv::Point(bbox.box.x2, bbox.box.y2), cv::Scalar(0, 255, 0), 2);

        // Prepare text for the label
        std::string label = "Class " + std::to_string(bbox.label) + " : " + std::to_string(bbox.probability);

        // Calculate text size
        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw rectangle for label background
        cv::rectangle(image, cv::Point(bbox.box.x1, bbox.box.y1- labelSize.height - baseline), cv::Point(bbox.box.x1 + labelSize.width, bbox.box.y1), cv::Scalar(0, 255, 0), cv::FILLED);

        // Put text on the image
        cv::putText(image, label, cv::Point(bbox.box.x1, bbox.box.y1 - baseline), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void VisionContainer::init() {
    LOG_INFO("Init Vision container called");
    Yolo yolo("yolov8n.onnx");
    // ONNXRuntimeInferenceEngine engine;
    // auto engine = InferenceEngineFactory::create_inference_engine();
    
    // LOG_INFO("Loading model");
    // engine->load_model("yolov8n.onnx");
    // LOG_INFO("Model loaded");
    cv::Mat mat = cv::imread("bus.jpg");
    Tensor<float> t = yolo.preprocess(mat);
    drawBoundingBoxes(mat, yolo.predict(t, mat));
    cv::imwrite("output.png", mat);
    // engine.run_inference(mat);
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
