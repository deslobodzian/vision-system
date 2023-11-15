#include "inference/onnxruntime_inference_engine.hpp"
#include "utils/logger.hpp"

std::string tensor_shape_to_string(const std::vector<int64_t>& input_tensor) {
    std::string tensor_shape = "Tensor shape is [";
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        tensor_shape += std::to_string(input_tensor[i]);
        if (i < input_tensor.size() - 1) {
            tensor_shape += ", ";
        }
    }
    tensor_shape += "]";
    return tensor_shape;
}


ONNXRuntimeInferenceEngine::ONNXRuntimeInferenceEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "Inference") {}

void ONNXRuntimeInferenceEngine::load_model_implementation(const std::string& model_path) {
    Tensor<float> t({1, 32, 32}, Device::CPU);
    std::vector<int64_t> temp = t.get_shape();
    LOG_INFO(tensor_shape_to_string(t.get_shape()));
    LOG_INFO(t);

    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1); // just for now 1 thread
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // might as well optimize :)

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    size_t input_node_size = session_.GetInputCount();
    size_t output_node_size = session_.GetOutputCount();

    printf("Input node size: %zu\n", input_node_size);
    printf("Output node size: %zu\n", output_node_size);

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    LOG_INFO("Input name: ", input_name);
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    LOG_INFO(tensor_shape_to_string(input_dims));
    auto output_name = session_.GetOutputNameAllocated(0, allocator_);
    LOG_INFO("Output name: ", output_name);

    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    LOG_INFO(tensor_shape_to_string(output_dims));

}

std::vector<float> ONNXRuntimeInferenceEngine::run_inference_implementation(const cv::Mat& image) {
    std::vector<float> tmp;
    return tmp;
}
