#include "inference/yolo.hpp"
#include "inference/cuda_utils.h"

#include <chrono>


Yolo::Yolo(const std::string& model_path) : model_path_(model_path) {
    inference_engine_ = InferenceEngineFactory::create_inference_engine();
    LOG_INFO("Loading Model: ", model_path_);
    inference_engine_->load_model(model_path_);
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
#endif
};

#ifdef WITH_CUDA
Tensor<float> Yolo::preprocess(const sl::Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
    preprocess_sl(image, inference_engine_->get_input_tensor(), stream_);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli>elapsed = stop - start;
    LOG_INFO("Preprocess took: ", elapsed.count());
    return Tensor<float>();
}
#endif

Tensor<float> Yolo::preprocess(const cv::Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
#ifdef WITH_CUDA
    preprocess_cv(image, inference_engine_->get_input_tensor(), stream_);
#else
    float scale = std::min(
        static_cast<float>(input_w_) / image.cols,
        static_cast<float>(input_h_) / image.rows
    );

    int scaled_w = static_cast<int>(image.cols * scale);
    int scaled_h = static_cast<int>(image.rows * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat output(input_w_, input_h_, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Rect roi((input_w_ - scaled_w) / 2, (input_h_ - scaled_h) / 2, scaled_w, scaled_h);
    resized.copyTo(output(roi));

    inference_engine_->get_input_tensor().to_cpu();
    inference_engine_->get_input_tensor().copy(output.data, Shape{output.cols, output.rows, output.channels()});
    LOG_INFO(inference_engine_->get_input_tensor().print_shape());

    // CWH -> WHC
    inference_engine_->get_input_tensor().scale(1.f / 255.f); // remember to normalize!!! I should add this to Tensor such that its .normalize()
    inference_engine_->get_input_tensor().reshape({1, 3, input_h_, input_w_});
#endif 
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli>elapsed = stop - start;
    LOG_INFO("Preprocess took: ", elapsed.count());
    return Tensor<float>();
}

std::vector<BBoxInfo> Yolo::postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<BBoxInfo> binfo;

    const float confidence_threshold = 0.5;
    const float thres = 0.8f;

    int image_w = image.cols;
    int image_h = image.rows;
    float scalingFactor = std::min(static_cast<float> (input_w_) / image_w, static_cast<float> (input_h_) / image_h);
    float xOffset = (input_w_ - scalingFactor * image_w) * 0.5f;
    float yOffset = (input_h_ - scalingFactor * image_h) * 0.5f;
    scalingFactor = 1.f / scalingFactor;
    float scalingFactor_x = scalingFactor;
    float scalingFactor_y = scalingFactor;

    auto num_channels = num_classes_ + bbox_values_;
    auto num_labels = num_classes_;

    auto& dw = xOffset;
    auto& dh = yOffset;

    auto& width = image_w;
    auto& height = image_h;
    LOG_INFO(prediction_tensor.print_shape());

    cv::Mat output = cv::Mat(
            num_channels,
            num_anchors_,
            CV_32F,
            static_cast<float*> (prediction_tensor.data())
            );

    output = output.t();
    cv::imwrite("tensor.png", output);
    for (int i = 0; i < num_anchors_; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + bbox_values_;
        auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score = *max_s_ptr;
        if (score > thres) {
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

            if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) break;

            bbi.label = label;
            bbi.probability = score;

            binfo.push_back(bbi);
        }
    }

    binfo = non_maximum_suppression(0.8, binfo);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli>elapsed = stop - start;
    LOG_INFO("Postprocess took: ", elapsed.count());
    return binfo;
}

std::vector<BBoxInfo> Yolo::predict(const cv::Mat& image) {
    Tensor<float> input = preprocess(image);
    inference_engine_->run_inference();
    return postprocess(inference_engine_->get_output_tensor(), image);
}
#ifdef WITH_CUDA
std::vector<BBoxInfo> Yolo::postprocess(const Tensor<float>& prediction_tensor, const sl::Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<BBoxInfo> binfo;

    const float confidence_threshold = 0.5;
    const float thres = 0.5f;

    int image_w = image.getWidth();
    int image_h = image.getHeight();
    float scalingFactor = std::min(static_cast<float> (input_w_) / image_w, static_cast<float> (input_h_) / image_h);
    float xOffset = (input_w_ - scalingFactor * image_w) * 0.5f;
    float yOffset = (input_h_ - scalingFactor * image_h) * 0.5f;
    scalingFactor = 1.f / scalingFactor;
    float scalingFactor_x = scalingFactor;
    float scalingFactor_y = scalingFactor;

    auto num_channels = num_classes_ + bbox_values_;
    auto num_labels = num_classes_;

    auto& dw = xOffset;
    auto& dh = yOffset;

    auto& width = image_w;
    auto& height = image_h;
    LOG_INFO(prediction_tensor.print_shape());

    cv::Mat output = cv::Mat(
            num_channels,
            num_anchors_,
            CV_32F,
            static_cast<float*> (prediction_tensor.data())
            );

    output = output.t();
    for (int i = 0; i < num_anchors_; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + bbox_values_;
        auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score = *max_s_ptr;
        if (score > thres) {
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

            if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) break;

            bbi.label = label;
            bbi.probability = score;

            binfo.push_back(bbi);
        }
    }

    binfo = non_maximum_suppression(0.5, binfo);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli>elapsed = stop - start;
    LOG_DEBUG("Detection objects: ", binfo.size());
    LOG_INFO("Postprocess took: ", elapsed.count());
    return binfo;
}

std::vector<BBoxInfo> Yolo::predict(const sl::Mat& image) {
    Tensor<float> input = preprocess(image);
    inference_engine_->run_inference();
    return postprocess(inference_engine_->get_output_tensor(), image);
}

cudaStream_t& Yolo::get_cuda_stream() {
    return stream_;
}

#endif
