#ifndef VISION_SYSTEM_POSTPROCESS_KERNELS_H
#define VISION_SYSTEM_POSTPROCESS_KERNELS_H

#include "inference/bbox.hpp"
#include "inference/cuda_utils.h"
#include "inference/tensor.hpp"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

struct CudaBBoxInfo {
  float x1, y1, x2, y2; // Coordinates
  int label;            // Class label
  float score;          // Confidence score
  int keep;             // Flag indicating whether to keep this box
};

void init_postprocess_resources(int num_anchors);

void postprocess(const Tensor<float> &prediction, std::vector<BBoxInfo> &bboxs,
                 const sl::Mat &img, int input_w, int input_h, float obj_thres,
                 float nms_thres);

#endif /* VISION_SYSTEM_POSTPROCESS_KERNELS_H */
