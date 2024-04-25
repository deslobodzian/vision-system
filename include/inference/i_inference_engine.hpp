#ifndef VISION_SYSTEM_I_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_I_INFERENCE_ENGINE_HPP

#include "tensor.hpp"
#include <string>

class IInferenceEngine {
public:
  virtual void load_model(const std::string &model_path) = 0;
  virtual void run_inference() = 0;

  virtual Tensor<float> &get_input_tensor() = 0;
  virtual Tensor<float> &get_output_tensor() = 0;
  virtual const Shape get_input_shape() const = 0;
  virtual const Shape get_output_shape() const = 0;
  virtual void set_execution_data(void *execution_data) {};

  virtual ~IInferenceEngine() = default;
};

#endif /* VISION_SYSTEM_I_INFERENCE_ENGINE_HPP */
