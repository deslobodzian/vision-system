//
// Created by deslobodzian on 5/2/22.
//

#ifndef VISION_SYSTEM_INFERENCE_MANAGER_HPP
#define VISION_SYSTEM_INFERENCE_MANAGER_HPP
//
#include <vector>
#include "yolov7.hpp"
#include "vision/Zed.hpp"

struct ZedInferenceStruct {
    sl::Mat sl_mat;
    cv::Mat cv_mat;
    std::vector<sl::CustomBoxObjectData> custom_obj_data_;
} ;

class InferenceManager {
public:
    InferenceManager(const std::string& custom_engine);
    ~InferenceManager();

    void init();
    void inference_on_device(Zed* camera);

private:
//    std::vector<std::thread> threads_;
    std::string engine_name_;
    Yolov7 detector_;
    ZedInferenceStruct zed_struct_;
};

#endif //VISION_SYSTEM_INFERENCE_MANAGER_HPP
