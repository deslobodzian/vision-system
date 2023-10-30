#ifndef VISION_SYSTEM_PREPROCESS_H
#define VISION_SYSTEM_PREPROCESS_H

#include <sl/Camera.hpp>
#include <cassert>
#include "cuda_utils.h"
#include <cuda_runtime.h>

void preprocess(const sl::Mat& left_img, float* d_input, int input_width, int input_height, size_t frame_s, int batch, cudaStream_t& stream);
void init_preprocess_resources(int image_width, int image_height, int input_width, int input_height); 
void free_preprocess_resources();


#endif // VISION_SYSTEM_PREPROCESS_H
