#ifndef VISION_SYSTEM_MODEL_FACTORY_HPP
#define VISION_SYSTEM_MODEL_FACTORY_HPP

#include "i_model.hpp"
#include "onnxruntime_inference_engine.hpp"
#include "tensor_rt_engine.hpp"

public:
static std::unique_ptr<IModel> create_inference_engine() {
  // for now only have yolo but can add more like rcnn
  return
}

private:
}
;

#endif /* VISION_SYSTEM_MODEL_FACTORY_HPP */
