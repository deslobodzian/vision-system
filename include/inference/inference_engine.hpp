#ifndef VISION_SYSTEM_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_INFERENCE_ENGINE_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

template<typename Implementation>
class InferenceEngine{
public:
    void load_model(const std::string& model_path) {
        static_cast<Implementation*>(this)->load_model_implementation(model_path);
    }

    /* 
    Need to figure out how to handle different inputs cv::mat vs sl::mat vs eigen::mat etc.
    For now assume all CPU and use opencv Mat
    */
    std::vector<float> run_inference(const cv::Mat& image) {
        return static_cast<Implementation*>(this)->run_model_implementation(image);
    }

    ~InferenceEngine() = default;
private:
    void load_model_implementation(const std::string& model_path);
    std::vector<float> run_inference_implementation(const cv::Mat& image);

};

#endif /* VISION_SYSTEM_INFERENCE_ENGINE_HPP */