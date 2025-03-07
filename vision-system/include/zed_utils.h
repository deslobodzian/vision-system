#ifdef CUDA
#pragma once

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

static int get_cv_type(sl::MAT_TYPE type) {
    int cv_type = -1;

    switch (type) {
        case sl::MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
    }
    return cv_type;
}

inline cv::Mat sl_to_cv(const sl::Mat& input) {
    return {
        static_cast<int>(input.getHeight()),
        static_cast<int>(input.getWidth()),
        get_cv_type(input.getDataType()),
        input.getPtr<sl::uchar1>(sl::MEM::CPU)
    };
}

#endif /* CUDA */