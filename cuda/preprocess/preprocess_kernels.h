#ifndef VISION_SYSTEM_PREPROCESS_KERNELS_H
#define VISION_SYSTEM_PREPROCESS_KERNELS_H

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cuda_runtime.h>
#include "inference/cuda_utils.h"
#include "inference/tensor.hpp"
#include "cuAprilTags.h"

void preprocess_sl(const sl::Mat& left_img, Tensor<float>& d_input, cudaStream_t& stream);
void preprocess_cv(const cv::Mat& img, Tensor<float>& d_input, cudaStream_t& stream);
void convert_sl_mat_to_april_tag_input(const sl::Mat& zed_mat, cuAprilTagsImageInput_t& tag_input, cudaStream_t stream, bool is_cam_0);

void init_april_tag_resources(int image_width, int image_height);
void init_preprocess_resources(int image_width, int image_height, int input_width, int input_height); 
void free_preprocess_resources();
void free_april_tag_resources();

#endif /* VISION_SYSTEM_PREPROCESS_KERNELS_H */
