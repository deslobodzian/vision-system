#include "postprocess_kernel.h"

// Based of off TensorRTX implementation, never tested
static CudaBBoxInfo* d_bbox_infos = nullptr;
static int* d_valid_count = nullptr;

static __device__ float clamp(float val, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(val, maxVal));
}

static __device__ float box_iou(CudaBBoxInfo a, CudaBBoxInfo b) {
    float c_left = max(a.x1, b.x1);
    float c_top = max(a.y1, b.y1);
    float c_right = min(a.x2, b.x2);
    float c_bottom = min(a.y2, b.y2);
    float c_area = max(0.0f, c_right - c_left) * max(0.0f, c_bottom - c_top);
    if (c_area == 0.0f) return 0.0f;

    float a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    float b_area = (b.x2 - b.x1) * (b.y2 - b.y1);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(CudaBBoxInfo *bbox_infos, int num_boxes, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    CudaBBoxInfo currentBox = bbox_infos[idx];
    if (currentBox.keep == 0) return; // Already marked for discard

    for (int i = 0; i < num_boxes; ++i) {
        if (i == idx) continue; // Skip self
        CudaBBoxInfo otherBox = bbox_infos[i];
        
        if (currentBox.label == otherBox.label) {
            float iou = box_iou(currentBox, otherBox);
            if (iou > nms_threshold) {
                if (currentBox.score >= otherBox.score) {
                    bbox_infos[i].keep = 0; 
                } else {
                    currentBox.keep = 0; 
                    bbox_infos[idx] = currentBox;
                    return;
                }
            }
        }
    }
}

static __global__ void postprocess_kernel(
        float *predictions, int num_anchors, int num_classes,
        float obj_threshold, float xOffset, float yOffset,
        float scalingFactor_x, float scalingFactor_y,
        int width, int height,
        CudaBBoxInfo *bbox_infos, int *valid_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    float *bboxes_ptr = predictions + idx * (4 + num_classes); 
    float *scores_ptr = bboxes_ptr + 4;

    float max_score = -1.0f;
    int best_class = -1;
    for (int i = 0; i < num_classes; ++i) {
        if (scores_ptr[i] > max_score) {
            max_score = scores_ptr[i];
            best_class = i;
        }
    }

    if (max_score > obj_threshold) {
        float x = bboxes_ptr[0] - xOffset;
        float y = bboxes_ptr[1] - yOffset;
        float w = bboxes_ptr[2];
        float h = bboxes_ptr[3];

        float x0 = clamp((x - 0.5f * w) * scalingFactor_x, 0.f, static_cast<float>(width));
        float y0 = clamp((y - 0.5f * h) * scalingFactor_y, 0.f, static_cast<float>(height));
        float x1 = clamp((x + 0.5f * w) * scalingFactor_x, 0.f, static_cast<float>(width));
        float y1 = clamp((y + 0.5f * h) * scalingFactor_y, 0.f, static_cast<float>(height));

        int index = atomicAdd(valid_count, 1);
        if (index < num_anchors) {
            CudaBBoxInfo& info = bbox_infos[index];
            info.x1 = x0;
            info.y1 = y0;
            info.x2 = x1;
            info.y2 = y1;
            info.label = best_class;
            info.score = max_score;
            info.keep = 1; // Initialize keep flag
        }
    }
}


void init_postprocess_resources(int num_anchors) {
    cudaMalloc(&d_bbox_infos, num_anchors * sizeof(CudaBBoxInfo));
    cudaMalloc(&d_valid_count, sizeof(int));
    cudaMemset(d_valid_count, 0, sizeof(int));

}

void postprocess(
        const Tensor<float>& prediction, 
        std::vector<BBoxInfo>& bboxs, 
        const sl::Mat& img, 
        int input_w, int input_h, 
        float obj_thres, float nms_thres) {
    Shape s = prediction.shape();
    int num_classes = s[1] - 4;
    int num_anchors = s[2]; // [batch, classes + bbox, anchors]

    int image_w = img.getWidth();
    int image_h = img.getHeight();

    float scaling_factor = std::min(
            static_cast<float>(input_w) / image_w,
            static_cast<float>(input_h) / image_h
    );

    float x_offset = (input_w - scaling_factor * image_w) * 0.5f;
    float y_offset = (input_h - scaling_factor * image_h) * 0.5f;
    scaling_factor = 1.f / scaling_factor;
    float scaling_factor_x = scaling_factor;
    float scaling_factor_y = scaling_factor;

    dim3 block(256);
    dim3 grid((num_anchors + block.x - 1) / block.x);

    postprocess_kernel<<<grid, block>>>(
            prediction.data(),
            num_anchors, num_classes,
            obj_thres,
            x_offset, y_offset,
            scaling_factor_x, scaling_factor_y,
            image_w, image_h,
            d_bbox_infos, d_valid_count
    );
    nms_kernel<<<grid, block>>>(d_bbox_infos, num_anchors, nms_thres);   
}

