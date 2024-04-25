#ifndef VISION_SYSTEM_INFERENCE_ENGINE_FACTORY_HPP
#define VISION_SYSTEM_INFERENCE_ENGINE_FACTORY_HPP

#include "i_inference_engine.hpp"
#include "onnxruntime_inference_engine.hpp"
#include "tensor_rt_engine.hpp"

class InferenceEngineFactory {
public:
  static std::unique_ptr<IInferenceEngine> create_inference_engine() {
    // prioritize cuda over cpu onnx
#ifdef WITH_CUDA
    return std::make_unique<TensorRTEngine>();
#endif
    return std::make_unique<ONNXRuntimeInferenceEngine>();
  }

private:
};

#endif /* VISION_SYSTEM_INFERENCE_ENGINE_FACTORY_HPP */
