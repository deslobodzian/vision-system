//
// Created by DSlobodzian on 1/2/2022.
//
#ifndef VISION_SYSTEM_YOLOV7_HPP
#define VISION_SYSTEM_YOLOV7_HPP

#include "config.h"
#include "model.h"
//#include <iostream>
//#include <chrono>
//#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "preprocess.h"
#include "postprocess.h"
//#include "common.hpp"
#include "utils.h"
#include "utils/utils.hpp"
#include <fstream>
//#include "utils.hpp"
//#include "calibrator.h"
//#include <mutex>
//
//#include "vision/monocular_camera.hpp"
#include <sl/Camera.hpp>
//
//#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
//#define DEVICE 0  // GPU id
//#define NMS_THRESH 0.4
//#define CONF_THRESH 0.5
//#define BATCH_SIZE 1
//
using namespace nvinfer1;

class Yolov7 {
//
private:
    ICudaEngine* engine_;
    IRuntime* runtime_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    void* buffers_[2];
    int inputIndex_;
    int outputIndex_;
    int batch_ = 0;
//    std::vector<sl::CustomBoxObjectData> objects_in_;
//
//
public:
    const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    float data[kBatchSize * 3 * kInputH * kInputW];
    float prob[kBatchSize * kOutputSize];

    Yolov7() = default;
    ~Yolov7();
//
    Logger gLogger;

//    int get_width(int x, float gw, int divisor = 8);
//
//    int get_depth(int x, float gd);
//
    bool initialize_engine(std::string& engine_name);
    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize);

//    std::vector<sl::uint2> cvt(const cv::Rect &bbox_in);
//    void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix);
    bool prepare_inference(cv::Mat& img_cv_rgb);
    bool prepare_inference(sl::Mat& img_sl, cv::Mat& img_cv_rgb);
//    void run_inference_and_convert_to_zed(cv::Mat& img_cv_rgb);
//    void run_inference(cv::Mat& img_cv_rgb);
//    std::vector<sl::CustomBoxObjectData> get_custom_obj_data();
//
//    void kill();
//
};
#endif // VISION_SYSTEM_YOLOV7_HPP
