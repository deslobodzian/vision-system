#ifndef VISION_SYSTEM_DETECTION_UTILS_HPP
#define VISION_SYSTEM_DETECTION_UTILS_HPP

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "inference/bbox.hpp"

#ifdef WITH_CUDA
inline std::vector<sl::uint2> cvt(const BBox &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

inline cv::Rect cvt_rect(const BBox &box) {
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

inline cv::Rect sl_cvt_rect(const std::vector<sl::uint2> &box) {
    return cv::Rect(round(box[0].x), round(box[0].y), round(box[2].x - box[0].x), round(box[2].y - box[0].y));
}

inline std::vector<sl::uint2> rect_to_sl(const cv::Rect& rect_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(rect_in.x, rect_in.y);
    bbox_out[1] = sl::uint2(rect_in.x + rect_in.width, rect_in.y);
    bbox_out[2] = sl::uint2(rect_in.x + rect_in.width, rect_in.y + rect_in.height);
    bbox_out[3] = sl::uint2(rect_in.x, rect_in.y + rect_in.height);
    return bbox_out;
}

#endif /* WITH_CUDA */

#endif /* VISION_SYSTEM_DETECTION_UTILS_HPP */
