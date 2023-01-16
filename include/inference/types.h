//
// Created by robot on 1/16/23.
//

#ifndef VISION_SYSTEM_TYPES_H
#define VISION_SYSTEM_TYPES_H
#include "config.h"

struct YoloKernel {
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};
#endif //VISION_SYSTEM_TYPES_H
