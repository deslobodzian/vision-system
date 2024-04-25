#ifndef VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP

#include "i_inference_engine.hpp"
#include "tensor.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>

class ONNXRuntimeInferenceEngine : public IInferenceEngine {
 public:
  ONNXRuntimeInferenceEngine();
  ~ONNXRuntimeInferenceEngine() = default;
  void load_model(const std::string& model_path) override;
  void run_inference() override;

  Tensor<float>& get_output_tensor() override;
  Tensor<float>& get_input_tensor() override;
  const Shape get_input_shape() const override;
  const Shape get_output_shape() const override;

 private:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_info_{nullptr};

  std::string input_node_name_;
  Shape input_shape_;

  std::string output_node_name_;
  Shape output_shape_;

  std::vector<int64_t> input_node_dims_;
  std::vector<int64_t> output_node_dims_;

  Tensor<float> input_;
  Tensor<float> output_;
};

#endif /* VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP */
