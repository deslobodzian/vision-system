#ifndef VISION_SYSTEM_I_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_I_INFERENCE_ENGINE_HPP

#include <string>
#include "tensor.hpp"

class IInferenceEngine {
public:
    virtual void load_model(const std::string& model_path) = 0;
    virtual Tensor<float> run_inference(const Tensor<float>& input_tensor) = 0;
    virtual const Shape get_input_shape() const = 0; 
    virtual const Shape get_output_shape() const = 0; 
    virtual ~IInferenceEngine() = default;
};

#endif /* VISION_SYSTEM_I_INFERENCE_ENGINE_HPP */
