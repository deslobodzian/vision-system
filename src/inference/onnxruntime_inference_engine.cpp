#include "inference/onnxruntime_inference_engine.hpp"
#include "inference/tensor_factory.hpp"
#include "utils/logger.hpp"
#include "utils/timer.h"
#include "utils/utils.hpp"
#include <opencv2/imgcodecs.hpp>

ONNXRuntimeInferenceEngine::ONNXRuntimeInferenceEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "Inference") {}

void ONNXRuntimeInferenceEngine::load_model(const std::string &model_path) {
  std::string onnx_model = remove_file_extension(model_path) + ".onnx";
  memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                            OrtMemType::OrtMemTypeDefault);

  std::vector<std::string> providers = Ort::GetAvailableProviders();
  for (auto provider : providers) {
    LOG_DEBUG("Provider available: ", provider);
  }

  Ort::SessionOptions session_options;
  session_options.SetInterOpNumThreads(1); // just for now 1 thread
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // might as well optimize :)

  session_ = Ort::Session(env_, onnx_model.c_str(), session_options);

  size_t input_node_size = session_.GetInputCount();
  size_t output_node_size = session_.GetOutputCount();

  printf("Input node size: %zu\n", input_node_size);
  printf("Output node size: %zu\n", output_node_size);

  input_node_name_ = session_.GetInputNameAllocated(0, allocator_).get();
  LOG_INFO("Input name: ", input_node_name_);
  Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_dims = input_tensor_info.GetShape();
  input_shape_ = input_tensor_info.GetShape();
  input_ = Tensor<float>(input_shape_, Device::CPU);
  LOG_INFO(input_.print_shape());

  output_node_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
  LOG_INFO("Output name: ", output_node_name_);

  Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  output_shape_ = output_tensor_info.GetShape();
  output_ = Tensor<float>(output_shape_, Device::CPU);
  LOG_INFO(output_.print_shape());
}

void ONNXRuntimeInferenceEngine::run_inference() {
  LOG_DEBUG("Running Inference");
  Timer t;
  // For now batchsize is always 1, so this doesn't need to be an array, but in
  // the future this might change.
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(
      TensorFactory<float>::to_ort_value(input_, memory_info_));

  const char *input_node_name_cstr = input_node_name_.c_str();
  const char *output_node_name_cstr = output_node_name_.c_str();

  t.start();
  std::vector<Ort::Value> output_tensor = session_.Run(
      Ort::RunOptions{nullptr}, &input_node_name_cstr, input_tensors.data(),
      1, // batch size of 1
      &output_node_name_cstr, 1);
  LOG_INFO("Inference took {", t.get_ms(), "} ms");

  output_ = TensorFactory<float>::from_ort_value(output_tensor.at(0));
}

Tensor<float> &ONNXRuntimeInferenceEngine::get_output_tensor() {
  return output_;
}

Tensor<float> &ONNXRuntimeInferenceEngine::get_input_tensor() { return input_; }

const Shape ONNXRuntimeInferenceEngine::get_input_shape() const {
  return input_shape_;
}

const Shape ONNXRuntimeInferenceEngine::get_output_shape() const {
  return output_shape_;
}
