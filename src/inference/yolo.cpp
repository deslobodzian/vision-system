<<<<<<< HEAD
#include "inference/yolo.hpp"
#include "inference/inference_engine_factory.hpp"

Yolo::Yolo(const std::string &model) : model_(model) {
  inference_engine_ = InferenceEngineFactory::create_inference_engine();
  LOG_INFO("Loading Model: ", model_);
  inference_engine_->load_model(model_);
  LOG_INFO("Model Loaded");
  Shape in = inference_engine_->get_input_shape();
  Shape out = inference_engine_->get_output_shape();

  input_w_ = in.at(2);
  input_h_ = in.at(3);

  bbox_values_ = 4;
  num_classes_ = out.at(1) - bbox_values_;
  num_anchors_ = out.at(2);
#ifdef WITH_CUDA
  LOG_INFO("Creating yolo cuda stream");
  CUDA_CHECK(cudaStreamCreate(&stream_));
  //inference_engine_.set_execution_data(static_cast<void *>(stream_));
#endif
};

void Yolo::configure(const detection_config &cfg) { cfg_ = cfg; }
=======
>>>>>>> ae539dd (Start creating new way for object detection)

