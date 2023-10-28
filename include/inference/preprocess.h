#ifndef VISION_SYSTEM_CONVERT_H
#define VISION_SYSTEM_CONVERT_H

#include <cuda_runtime.h>
#include <sl/Camera.hpp>
#include <cassert>
#include "cuda_utils.h"

void preprocess(sl::Mat& left_img, float* d_input, int input_width, int input_height, size_t frame_s, int batch, cudaStream_t& stream);

#endif // VISION_SYSTEM_CONVERT_H
