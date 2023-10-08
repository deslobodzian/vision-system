//
// Created by deslobodzian on 5/2/22.
//

#ifndef VISION_SYSTEM_INFERENCE_MANAGER_HPP
#define VISION_SYSTEM_INFERENCE_MANAGER_HPP
//
#include <vector>
#include <string>
// #include "yolov7.hpp"
#include "yolo.hpp"
#include "vision/Zed.hpp"
#include "inference_utils.hpp"

#define NMS_THRESH 0.4
#define CONF_THRESH 0.3

struct ZedInferenceStruct {
    sl::Mat sl_mat;
    cv::Mat cv_mat;
    std::vector<sl::CustomBoxObjectData> custom_obj_data_;
};

class InferenceManager {
public:
    InferenceManager(const std::string& custom_engine);
    ~InferenceManager();
    void init(int camera_resolution_h, int camera_resolution_w);
    void inference_on_device(Zed* camera);
//     void test_inference(cv::Mat& img);

private:
// //    std::vector<std::thread> threads_;
    std::string engine_name_;
    Yolo detector_;
//     Yolov7 detector_;
    ZedInferenceStruct zed_struct_;
    int camera_resolution_h_, camera_resolution_w_;
};

#endif //VISION_SYSTEM_INFERENCE_MANAGER_HPP
