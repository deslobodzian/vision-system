#ifndef VISION_SYSTEM_YOLO_HPP 
#define VISION_SYSTEM_YOLO_HPP 

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils/logger.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include "inference_engine_factory.hpp"
#include "bbox.hpp"
#include "postprocess.hpp"

#ifdef WITH_CUDA
#include "preprocess_kernels.h"
#endif

class Yolo {
public:
    Yolo(const std::string& model_path);
    ~Yolo() = default;
    Tensor<float> preprocess(const cv::Mat& image);
    std::vector<BBoxInfo> postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image);
    std::vector<BBoxInfo> predict(const cv::Mat& image);
private:
    std::unique_ptr<IInferenceEngine> inference_engine_;

    std::string model_path_;

    int input_h_;
    int input_w_;

    int bbox_values_;
    int num_classes_;
    int num_anchors_;
    cudaStream_t stream_;

};

#endif /* VISION_SYSTEM_YOLO_HPP */
