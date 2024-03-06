#ifndef VISION_SYSTEM_PREPROCESS_KERNELS_H
#define VISION_SYSTEM_PREPROCESS_KERNELS_H

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "inference/cuda_utils.h"
#include "inference/tensor.hpp"
#include "cuAprilTags.h"

struct kernel_resources {
    unsigned char* d_bgr;
    uchar3* d_april_tag_bgr;
    unsigned char* d_output;
    int max_image_width;
    int max_image_height;
};

void preprocess_sl(const sl::Mat& left_img, Tensor<float>& d_input, kernel_resources& resources, cudaStream_t& stream);
void preprocess_cv(const cv::Mat& img, Tensor<float>& d_input, cudaStream_t& stream);

void convert_sl_mat_to_april_tag_input(const sl::Mat& zed_mat, cuAprilTagsImageInput_t& tag_input, kernel_resources& resources, cudaStream_t stream = 0);

void init_april_tag_resources(int image_width, int image_height);
void init_april_tag_resources(kernel_resources& resources, int image_width, int image_height);

void init_preprocess_resources(int image_width, int image_height, int input_width, int input_height); 

void free_preprocess_resources();
void free_april_tag_resources();

void free_preprocess_resources(kernel_resources& resources);
void free_april_tag_resources(kernel_resources& resources);

#endif /* VISION_SYSTEM_PREPROCESS_KERNELS_H */
