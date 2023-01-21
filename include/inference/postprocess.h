//
// Created by robot on 1/16/23.
//

#ifndef VISION_SYSTEM_POSTPROCESS_H
#define VISION_SYSTEM_POSTPROCESS_H
#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh = 0.5);

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);
void draw_bbox_single(cv::Mat& img, std::vector<Detection>& res);
#endif //VISION_SYSTEM_POSTPROCESS_H
