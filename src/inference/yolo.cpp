#include "inference/yolo.hpp"

Yolo::Yolo(const std::string& model_path) : model_path_(model_path) {
    inference_engine_ = InferenceEngineFactory::create_inference_engine();
    LOG_INFO("Loading Model: ", model_path_);
    inference_engine_->load_model(model_path_);
};

Tensor<float> Yolo::preprocess(const cv::Mat& image) {
    Shape input_shape = inference_engine_->get_input_shape();
    int input_w = input_shape.at(2);
    int input_h = input_shape.at(3);
    LOG_DEBUG("Input Width: ", input_w, " Input height: ", input_h);
    float scale = std::min(
        static_cast<float>(input_w) / image.cols,
        static_cast<float>(input_h) / image.rows
    );

    int scaled_w = static_cast<int>(image.cols * scale);
    int scaled_h = static_cast<int>(image.rows * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat output(input_w, input_h, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Rect roi((input_w - scaled_w) / 2, (input_h - scaled_h) / 2, scaled_w, scaled_h);
    resized.copyTo(output(roi));

    Tensor<unsigned char> tensor = TensorFactory<unsigned char>::from_cv_mat(output);
    LOG_INFO(tensor.print_shape());
    // CWH -> WHC
    Tensor<float> input_tensor = tensor;
    input_tensor.scale(1.f / 255.f); // remember to normalize!!! I should add this to Tensor such that its .normalize() but normize based on tensor type.
    input_tensor.permute({2, 0, 1});
    // I forgot the name in python but there is a function that added a dimension. I should add this
    input_tensor.reshape({1, 3, input_h, input_w});
    LOG_INFO("Input tensor shape: ", input_tensor.print_shape());
    return input_tensor;
}

std::vector<BBoxInfo> Yolo::postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image) {
    std::vector<BBoxInfo> binfo;

    const int input_width = 640;
    const int input_height = 640;
    const float confidence_threshold = 0.5;
    const int num_anchors = 8400;
    const int num_classes = 80; // Assuming 80 classes for YOLOv8
    const int bbox_values = 4;  // Number of values for bbox (x, y, width, height)
    const int num_features = bbox_values + num_classes; // Total features per anchor
    const float thres = 0.8f;

    int image_w = image.cols;
    int image_h = image.rows;
    float scalingFactor = std::min(static_cast<float> (input_width) / image_w, static_cast<float> (input_height) / image_h);
    float xOffset = (input_width - scalingFactor * image_w) * 0.5f;
    float yOffset = (input_height - scalingFactor * image_h) * 0.5f;
    scalingFactor = 1.f / scalingFactor;
    float scalingFactor_x = scalingFactor;
    float scalingFactor_y = scalingFactor;

    auto num_channels = num_classes + bbox_values;
    auto num_labels = num_classes;

    auto& dw = xOffset;
    auto& dh = yOffset;

    auto& width = image_w;
    auto& height = image_h;
    LOG_INFO(prediction_tensor.print_shape());

    cv::Mat output = cv::Mat(
            num_channels,
            num_anchors,
            CV_32F,
            static_cast<float*> (prediction_tensor.data())
            );
    output = output.t();
    cv::imwrite("tensor.png", output);
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + bbox_values;
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
    return binfo;
}
// std::vector<BBoxInfo> Yolo::postprocess(const Tensor<float>& prediction_tensor, const cv::Mat& image) {
//     const int input_width = 640;
//     const int input_height = 640;
//     const float confidence_threshold = 0.5;
//     const int num_anchors = 8400;
//     const int num_classes = 80; // Assuming 80 classes for YOLOv8
//     const int bbox_values = 4;  // Number of values for bbox (x, y, width, height)
//     const int num_features = bbox_values + num_classes; // Total features per anchor

//     int image_width = image.cols;
//     int image_height = image.rows;

//     // Check if the tensor shape is valid
//     if (prediction_tensor.shape().size() != 3 || prediction_tensor.shape()[0] != 1 || prediction_tensor.shape()[1] != num_features || prediction_tensor.shape()[2] != num_anchors) {
//         throw std::runtime_error("Invalid shape for YOLOv8 output tensor");
//     }

//     // Calculate scaling factors and offsets
//     float scaling_factor = std::min(static_cast<float>(input_width) / image_width, static_cast<float>(input_height) / image_height);
//     float x_offset = (input_width - scaling_factor * image_width) * 0.5f;
//     float y_offset = (input_height - scaling_factor * image_height) * 0.5f;

//     scaling_factor = 1.f / scaling_factor;
//     float scaling_factor_x = scaling_factor;
//     float scaling_factor_y = scaling_factor;

//     float* data = prediction_tensor.data();
//     std::vector<BBoxInfo> binfo;

//     // Decode bounding boxes and class probabilities
//     for (int i = 0; i < num_anchors; ++i) {
//         float* anchor_data = data + (i * num_features);

//         float bx = anchor_data[0] - x_offset;
//         float by = anchor_data[1] - y_offset;
//         float bw = anchor_data[2];
//         float bh = anchor_data[3];

//         // Adjust bounding box coordinates
//         float x = (bx - bw / 2) * scaling_factor_x;
//         float y = (by - bh / 2) * scaling_factor_y;
//         float x2 = (bx + bw / 2) * scaling_factor_x;
//         float y2 = (by + bh / 2) * scaling_factor_y;

//         // Clamp coordinates to image dimensions
//         x = clamp(x, 0, image_width);
//         y = clamp(y, 0, image_height);
//         x2 = clamp(x2, 0, image_width);
//         y2 = clamp(y2, 0, image_height);

//         // Find the class with the highest score
//         float max_class_score = 0;
//         int class_id = -1;
//         for (int j = bbox_values; j < num_features; ++j) {
//             float class_score = anchor_data[j];
//             if (class_score > max_class_score) {
//                 max_class_score = class_score;
//                 class_id = j - bbox_values;
//             }
//         }

//         // Filter out low confidence detections
//         if (max_class_score > confidence_threshold) {
//             BBoxInfo bbox;
//             bbox.box.x1 = x;
//             bbox.box.y1 = y;
//             bbox.box.x2 = x2;
//             bbox.box.y2 = y2;
//             bbox.label = class_id;
//             bbox.probability = max_class_score;

//             binfo.push_back(bbox);
//         }
//     }

//     // Apply non-maximum suppression
//     const float nms_threshold = 0.5;
//     std::vector<BBoxInfo> final_detections = weigthed_nms(nms_threshold, binfo);

//     return final_detections;
// }


std::vector<BBoxInfo> Yolo::predict(const Tensor<float>& input_tensor, const cv::Mat& image) {
    Tensor<float> output = inference_engine_->run_inference(input_tensor);
    float confidence_threshold = 0.5f; // Set your confidence threshold
    return postprocess(output, image);
}