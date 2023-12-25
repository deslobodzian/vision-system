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

class Yolo {
public:
    Yolo(const std::string& model_path);
    ~Yolo() = default;
    Tensor<float> preprocess(const cv::Mat& image);
    std::vector<BBoxInfo> postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image);
    std::vector<BBoxInfo> predict(const Tensor<float>& input_tensor, const cv::Mat& image);
private:
    std::unique_ptr<IInferenceEngine> inference_engine_;
    std::string model_path_;
};

#endif /* VISION_SYSTEM_YOLO_HPP */