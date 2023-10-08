//
// Created by deslobodzian on 5/2/22.
//

#include "inference/inference_manager.hpp"

InferenceManager::InferenceManager(const std::string& custom_engine) {
    engine_name_ = custom_engine;
}

void InferenceManager::init(int camera_resolution_h, int camera_resolution_w) {
    detector_.init(engine_name_);
    camera_resolution_h_ = camera_resolution_h;
    camera_resolution_w_ = camera_resolution_w;
}

/**
 * runs inference on device, make sure to fetch camera before calling.
 * @param camera
 */
void InferenceManager::inference_on_device(Zed *camera) {
    zed_struct_.custom_obj_data_.clear();
    camera->get_left_image(zed_struct_.sl_mat);
    auto detections = detector_.run(zed_struct_.sl_mat, camera_resolution_h_, camera_resolution_w_, CONF_THRESH);
    cv::Rect bounds = cv::Rect(0, 0, camera_resolution_w_, camera_resolution_h_);
    for (auto &it : detections) {
        sl::CustomBoxObjectData tmp;
        tmp.unique_object_id = sl::generate_unique_id();
        tmp.probability = it.prob;
        tmp.label = (int) it.label;
        cv::Rect r = cvt_rect(it.box);
        r = bounds & r;

        tmp.bounding_box_2d = rect_to_sl(r);
        tmp.is_grounded = ((int) it.label == 0); 
        // others are tracked in full 3D space                
        zed_struct_.custom_obj_data_.push_back(tmp);
    }
    camera->ingest_custom_objects(zed_struct_.custom_obj_data_);
}

// void InferenceManager::test_inference(cv::Mat& img) {
//     detector_.prepare_inference(img);
//     detector_.run_inference_test(img);
// }

InferenceManager::~InferenceManager() {
    // detector_.kill();
}

