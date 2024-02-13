#ifndef VISION_SYSTEM_BBOX_HPP
#define VISION_SYSTEM_BBOX_HPP
// maybe add conversions from differnt bbox types like xyxy to xywh
struct BBox {
    float x1, x2, y1, y2;
};

struct BBoxInfo{
    BBox box;
    int label;
    float probability;
};


#endif /* VISION_SYSTEM_BBOX_HPP */
