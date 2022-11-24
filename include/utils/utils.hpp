#ifndef VISION_SYSTEM_UTILS_HPP
#define VISION_SYSTEM_UTILS_HPP

#include <math.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sl/Camera.hpp>
#include <sophus/geometry.hpp>

inline void draw_vertical_line(
        cv::Mat &left_display,
        const cv::Point2i& start_pt,
        const cv::Point2i& end_pt,
        const cv::Scalar& clr,
        int thickness) {
    int n_steps = 7;
    cv::Point2i pt1, pt4;
    pt1.x = ((n_steps - 1) * start_pt.x + end_pt.x) / n_steps;
    pt1.y = ((n_steps - 1) * start_pt.y + end_pt.y) / n_steps;

    pt4.x = (start_pt.x + (n_steps - 1) * end_pt.x) / n_steps;
    pt4.y = (start_pt.y + (n_steps - 1) * end_pt.y) / n_steps;

    cv::line(left_display, start_pt, pt1, clr, thickness);
    cv::line(left_display, pt4, end_pt, clr, thickness);
}

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

inline cv::cuda::GpuMat slMat_to_cvMat_GPU(const sl::Mat& input) {
    return {
        (int)input.getHeight(),
        (int)input.getWidth(),
        get_ocv_type(input.getDataType()),
        input.getPtr<sl::uchar1>(sl::MEM::GPU),
        input.getStepBytes(sl::MEM::GPU)
    };
}

inline sl::float3 calculate_plane_normal_vector(const sl::float3 &p1, const sl::float3 &p2, const sl::float3 &p3) {
    sl::float3 u = sl::float3::cross((p3 - p1), (p2 - p1));
    return u/u.norm();
}

inline Sophus::SO3d so3_from_normal_vec(const sl::float3 &normal_vec) {
    return Sophus::SO3FromNormal(Eigen::Vector3d(normal_vec.x, normal_vec.y, normal_vec.z));
}

inline sl::Orientation orientation_from_normal_vec(const sl::float3 &normal_vec) {
    Sophus::SO3d so3(so3_from_normal_vec(normal_vec));
    Sophus::SO3d::QuaternionMember q = so3.unit_quaternion();
    return {sl::float4((float)q.x(), (float)q.y(), (float)q.z(), (float)q.w())};
}

inline bool is_vec_nan(const sl::float3 &vec) {
    return isnan(abs(vec.x)) || isnan(abs(vec.y)) || isnan(abs(vec.z));
}

inline std::string pose_string(sl::Pose &pose) {
    std::string tmp = "Pose {x: " + std::to_string(pose.getTranslation().tx) +
            ", y: " + std::to_string(pose.getTranslation().ty) +
            ", z: " + std::to_string(pose.getTranslation().tz) +
            "} angle: " + std::to_string(pose.getEulerAngles(false).z);
    return tmp;
}

inline void info(const std::string& msg) {
    std::cout << "[INFO] " << msg << "\n";
}
inline void error(const std::string& msg) {
    std::cout << "[ERROR] " << msg << "\n";
}
inline void debug(const std::string& msg) {
    std::cout << "[DEBUG] " << msg << "\n";
}
/*
 * Encode int32_t to uint8_t from LCM package
 */
inline int encode_int32_t_array(void *_buf, int offset, int maxlen, const int32_t *p, int elements) {
    int total_size = sizeof(int32_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    const auto *unsigned_p = (const uint32_t *) p;
    for (element = 0; element < elements; element++) {
        const uint32_t v = unsigned_p[element];
        buf[pos++] = (v >> 24) & 0xff;
        buf[pos++] = (v >> 16) & 0xff;
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
    }

    return total_size;
}
inline int encode_float_array(void *_buf, int offset, int maxlen, const float *p, int elements) {
    return encode_int32_t_array(_buf, offset, maxlen, reinterpret_cast<const int32_t *>(p), elements);
}
/*
 * decode uint8_t to uint8_t from LCM package.
 */
inline int decode_int32_t_array(
        const void *_buf,
        int offset,
        int maxlen,
        int32_t *p,
        int elements) {
    int total_size = sizeof(int32_t) * elements;
    const auto *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;
    if (maxlen < total_size)
        return -1;
    for (element = 0; element < elements; element++) {
        p[element] = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + ((uint32_t) buf[pos + 3]);
        pos += 4;
    }
    return total_size;
}

inline int decode_float_array(const void *_buf, int offset, int maxlen, float *p, int elements) {
    return decode_int32_t_array(_buf, offset, maxlen, reinterpret_cast<int32_t *>(p), elements);
}
inline int encode_int64_t_array(void *_buf, int offset, int maxlen, const int64_t *p,
                                         int elements)
{
    int total_size = sizeof(int64_t) * elements;
    uint8_t *buf = (uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    const uint64_t *unsigned_p = (const uint64_t *) p;
    for (element = 0; element < elements; element++) {
        const uint64_t v = unsigned_p[element];
        buf[pos++] = (v >> 56) & 0xff;
        buf[pos++] = (v >> 48) & 0xff;
        buf[pos++] = (v >> 40) & 0xff;
        buf[pos++] = (v >> 32) & 0xff;
        buf[pos++] = (v >> 24) & 0xff;
        buf[pos++] = (v >> 16) & 0xff;
        buf[pos++] = (v >> 8) & 0xff;
        buf[pos++] = (v & 0xff);
    }
    return total_size;
}

static inline int encode_double_array(void *_buf, int offset, int maxlen, const double *p, int elements) {
    return encode_int64_t_array(_buf, offset, maxlen, reinterpret_cast<const int64_t *>(p), elements);
}

static inline int decode_int64_t_array(const void *_buf, int offset, int maxlen, int64_t *p,
                                         int elements)
{
    int total_size = sizeof(int64_t) * elements;
    const uint8_t *buf = (const uint8_t *) _buf;
    int pos = offset;
    int element;

    if (maxlen < total_size)
        return -1;

    for (element = 0; element < elements; element++) {
        uint64_t a = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        uint64_t b = (((uint32_t) buf[pos + 0]) << 24) + (((uint32_t) buf[pos + 1]) << 16) +
                     (((uint32_t) buf[pos + 2]) << 8) + (uint32_t) buf[pos + 3];
        pos += 4;
        p[element] = (a << 32) + (b & 0xffffffff);
    }

    return total_size;
}

static inline int decode_double_array(const void *_buf, int offset, int maxlen, double *p, int elements) {
    return decode_int64_t_array(_buf, offset, maxlen, reinterpret_cast<int64_t *>(p), elements);
}


#endif
