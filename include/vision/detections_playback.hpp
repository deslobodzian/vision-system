#ifndef VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP
#define VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP

#include <cmath>
#include <string>
#include <sl/Camera.hpp>

#include "inference/yolo.hpp"
#include "zed.hpp"

#define NMS_THRESH 0.4
#define CONF_THRESH 0.3

static int get_ocv_type(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
            break;
        default: break;
    }
    return cv_type;
}

inline cv::Mat slMat_to_cvMat(const sl::Mat &input) {
    // Mapping between MAT_TYPE and CV_TYPE
    return {
        (int)input.getHeight(),
        (int)input.getWidth(),
        get_ocv_type(input.getDataType()),
        input.getPtr<sl::uchar1>(sl::MEM::CPU)
    };
}


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

inline std::vector<sl::uint2> rect_to_sl(const cv::Rect& rect_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(rect_in.x, rect_in.y);
    bbox_out[1] = sl::uint2(rect_in.x + rect_in.width, rect_in.y);
    bbox_out[2] = sl::uint2(rect_in.x + rect_in.width, rect_in.y + rect_in.height);
    bbox_out[3] = sl::uint2(rect_in.x, rect_in.y + rect_in.height);
    return bbox_out;
}


class DetectionsPlayback {
public:
    DetectionsPlayback(const std::string& svo_file);
    ~DetectionsPlayback();
    
    void detect();
    void export_video();
    
private:
    zed_config cfg;
    Yolo yolo_;
    //ZedCamera zed_;
    sl::Camera zed_;
    sl::Mat left_sl;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Resolution display_resolution;
    cv::Mat left_cv;
    cv::VideoWriter video_writer;
};

#endif /* VISION_SYSTEM_DETECTIONS_PLAYBACK_HPP */
