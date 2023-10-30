#ifndef VISION_SYSTEM_YOLO_HPP
#define VISION_SYSTEM_YOLO_HPP

#include <NvInfer.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "logging.h"
#include "cuda_utils.h"
#include "inference_utils.hpp"
#include "utils/utils.hpp"
#include "preprocess.h"

class Yolo {
    public:
        Yolo();
        ~Yolo();

        static int build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile);
        int init(std::string engine_name);
        void yolo_preprocess(const sl::Mat& img);
        std::vector<BBoxInfo> run(sl::Mat left_img, int orig_image_h, int orig_image_w, float thres);

        sl::Resolution getInferenceSize() {
            return sl::Resolution(input_width_, input_height_);
        }
    private:
        cv::Mat left_cv_rgb_;

        float nms_ = 0.4;

        std::string input_binding_name_ = "images";
        std::string output_name_ = "classes";
        int input_index_, output_index_;

        size_t input_width_ = 0, input_height_ = 0, batch_size_=1;

        size_t out_dim_ = 8400;
        size_t out_class_number_ = 80;
        size_t out_box_struct_number_ = 4;
        size_t output_size_ = 0;

        float *h_input_, *h_output_;
        float *d_input_, *d_output_;

        nvinfer1::IRuntime* runtime_;
        nvinfer1::ICudaEngine* engine_;
        nvinfer1::IExecutionContext* context_;
        cudaStream_t stream_;
        bool is_init_ = false;
};
#endif /* VISION_SYSTEM_YOLO_HPP */