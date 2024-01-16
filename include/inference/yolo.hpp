#ifndef VISION_SYSTEM_YOLO_HPP 
#define VISION_SYSTEM_YOLO_HPP 

#include <vector>
#include <opencv2/opencv.hpp>
#include "inference/i_inference_engine.hpp"
#include "tensor.hpp"
#include "bbox.hpp"

#ifdef WITH_CUDA
#include <sl/Camera.hpp>
#endif

struct detection_config {
    float nms_thres = 0.5;
    float obj_thres = 0.5;
    detection_config() = default;
};

class Yolo {
public:
    Yolo(const std::string& model);
    ~Yolo() = default;

    void configure(const detection_config& cfg);

    Tensor<float> preprocess(const cv::Mat& image);
    std::vector<BBoxInfo> postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image);
    std::vector<BBoxInfo> predict(const cv::Mat& image);

#ifdef WITH_CUDA
    Tensor<float> preprocess(const sl::Mat& image);
    std::vector<BBoxInfo> predict(const sl::Mat& image);
    std::vector<BBoxInfo> postprocess(const Tensor<float>& prediction_tensor, const sl::Mat& image);
#endif

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

#endif /* VISION_SYSTEM_YOLO_HPP */
