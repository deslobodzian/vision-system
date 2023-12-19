#ifndef VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP

#include <onnxruntime_cxx_api.h>
#include <string>
#include "inference_engine.hpp"
#include "tensor.hpp"

class ONNXRuntimeInferenceEngine : public InferenceEngine<ONNXRuntimeInferenceEngine> {
public:
    ONNXRuntimeInferenceEngine();
    ~ONNXRuntimeInferenceEngine() = default;
    void load_model_implementation(const std::string& model_path);
    std::vector<float> run_inference_implementation(const cv::Mat& image);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{nullptr};

    std::string input_node_name_;
    std::vector<int64_t> input_node_dims_;
    std::string output_node_name_;
    std::vector<int64_t> output_node_dims_;
};

#endif /* VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP */