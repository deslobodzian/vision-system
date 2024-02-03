#ifndef VISION_SYSTEM_YOLO_HPP 
#define VISION_SYSTEM_YOLO_HPP 

#include <vector>
#include <opencv2/opencv.hpp>
#include "i_inference_engine.hpp"
#include "tensor.hpp"
#include "bbox.hpp"
#include "inference/postprocess.hpp"
#include "utils/logger.hpp"
#include <chrono>
#include "i_model.hpp"
#include "inference_engine_factory.hpp"
#include "yolo_processing.hpp"

#ifdef WITH_CUDA
#include <sl/Camera.hpp>
#include "inference/cuda_utils.h"
#include "preprocess_kernels.h"
#endif

template <typename MatType>
class Yolo : public IModel<MatType> {
public:
    Yolo<MatType>(const std::string& model);
    ~Yolo() = default;

    void configure(const detection_config& cfg) override;
    void preprocess(const MatType& image) override;
    std::vector<BBoxInfo> predict(const MatType& image) override;
    std::vector<BBoxInfo> postprocess(const Tensor<float>& prediction_tensor, const MatType& image) override;

private:
    std::unique_ptr<IInferenceEngine> inference_engine_;
    std::string model_;
    detection_config cfg_;

    int input_h_;
    int input_w_;

    int bbox_values_;
    int num_classes_;
    int num_anchors_;

#ifdef WITH_CUDA
    cudaStream_t stream_;
#endif

};

template <typename MatType>
Yolo<MatType>::Yolo(const std::string &model) : model_(model) {
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
  inference_engine_->set_execution_data(stream_);
#endif
}

template <typename MatType>
void Yolo<MatType>::configure(const detection_config &cfg) {
  cfg_ = cfg;
}
template <typename MatType>
void Yolo<MatType>::preprocess(const MatType &image) {
  auto start = std::chrono::high_resolution_clock::now();

  if constexpr (std::is_same_v<MatType, cv::Mat>) {
#ifdef WITH_CUDA
    preprocess_cv(image, inference_engine_->get_input_tensor(), stream_);
#else
    cpu_preprocess(image, inference_engine_->get_input_tensor());
#endif
  } else {
#ifdef WITH_CUDA
    LOG_DEBUG("Using sl::Mat");
    preprocess_sl(image, inference_engine_->get_input_tensor(), stream_);
#else
    LOG_ERROR("No Cuda, use cpu cv::Mat");
#endif
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = stop - start;
  LOG_INFO("Preprocess took: ", elapsed.count());
}

template <typename MatType>
std::vector<BBoxInfo> Yolo<MatType>::postprocess(const Tensor<float> &prediction_tensor,
                                        const MatType &image) {
  int image_w;
  int image_h;
  if constexpr (std::is_same_v<MatType, cv::Mat>) {
    image_w = image.cols;
    image_h = image.rows;
  } else {
#ifdef WITH_CUDA
    image_w = image.getWidth();
    image_h = image.getHeight();
#endif
  }
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<BBoxInfo> binfo;

  float scalingFactor = std::min(static_cast<float>(input_w_) / image_w,
                                 static_cast<float>(input_h_) / image_h);
  float xOffset = (input_w_ - scalingFactor * image_w) * 0.5f;
  float yOffset = (input_h_ - scalingFactor * image_h) * 0.5f;
  scalingFactor = 1.f / scalingFactor;
  float scalingFactor_x = scalingFactor;
  float scalingFactor_y = scalingFactor;

  auto num_channels = num_classes_ + bbox_values_;
  auto num_labels = num_classes_;

  auto &dw = xOffset;
  auto &dh = yOffset;

  auto &width = image_w;
  auto &height = image_h;
  LOG_INFO(prediction_tensor.print_shape());

  cv::Mat output = cv::Mat(num_channels, num_anchors_, CV_32F,
                           static_cast<float *>(prediction_tensor.data()));

  output = output.t();
  for (int i = 0; i < num_anchors_; i++) {
    auto row_ptr = output.row(i).ptr<float>();
    auto bboxes_ptr = row_ptr;
    auto scores_ptr = row_ptr + bbox_values_;
    auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
    float score = *max_s_ptr;
    if (score > cfg_.obj_thres) {
      int label = max_s_ptr - scores_ptr;

      BBoxInfo bbi;

      float x = *bboxes_ptr++ - dw;
      float y = *bboxes_ptr++ - dh;
      float w = *bboxes_ptr++;
      float h = *bboxes_ptr;

      float x0 = clamp((x - 0.5f * w) * scalingFactor_x, 0.f, width);
      float y0 = clamp((y - 0.5f * h) * scalingFactor_y, 0.f, height);
      float x1 = clamp((x + 0.5f * w) * scalingFactor_x, 0.f, width);
      float y1 = clamp((y + 0.5f * h) * scalingFactor_y, 0.f, height);

      cv::Rect_<float> bbox;
      bbox.x = x0;
      bbox.y = y0;
      bbox.width = x1 - x0;
      bbox.height = y1 - y0;

      bbi.box.x1 = x0;
      bbi.box.y1 = y0;
      bbi.box.x2 = x1;
      bbi.box.y2 = y1;

      if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
        break;

      bbi.label = label;
      bbi.probability = score;

      binfo.push_back(bbi);
    }
  }

  binfo = non_maximum_suppression(cfg_.nms_thres, binfo);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = stop - start;
  LOG_INFO("Postprocess took: ", elapsed.count());
  return binfo;
}

template <typename MatType>
std::vector<BBoxInfo> Yolo<MatType>::predict(const MatType &image) {
  preprocess(image);
  inference_engine_->run_inference();
  return postprocess(inference_engine_->get_output_tensor(), image);
}

#endif /* VISION_SYSTEM_YOLO_HPP */
