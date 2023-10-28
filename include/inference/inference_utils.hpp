#pragma once

#include <NvInfer.h>
#include <vector>
#include <stdio.h>
#include <algorithm>

struct BBox {
    float x1, y1, x2, y2;
};

struct BBoxInfo {
    BBox box;
    int label;
    float prob;
};

inline std::vector<std::string> split_str(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

struct OptimDim {
    nvinfer1::Dims4 size;
    std::string tensor_name;

    bool setFromString(std::string &arg) {
        // "images:1x3x512x512"
        std::vector<std::string> v_ = split_str(arg, ":");
        if (v_.size() != 2) return true;

        std::string dims_str = v_.back();
        std::vector<std::string> v = split_str(dims_str, "x");

        size.nbDims = 4;
        // assuming batch is 1 and channel is 3
        size.d[0] = 1;
        size.d[1] = 3;

        if (v.size() == 2) {
            size.d[2] = stoi(v[0]);
            size.d[3] = stoi(v[1]);
        } else if (v.size() == 3) {
            size.d[2] = stoi(v[1]);
            size.d[3] = stoi(v[2]);
        } else if (v.size() == 4) {
            size.d[2] = stoi(v[2]);
            size.d[3] = stoi(v[3]);
        } else return true;

        if (size.d[2] != size.d[3]) std::cerr << "Warning only squared input are currently supported" << std::endl;

        tensor_name = v_.front();
        return false;
    }
};

inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    printf("x_off = %f, y_off = %f\n", x, y);
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

inline int clamp(int val, int min, int max) {
    if (val <= min) return min;
    if (val >= max) return max;
    return val;
}

#define WEIGHTED_NMS

inline std::vector<BBoxInfo> non_maximum_suppression(const float nms_thresh, std::vector<BBoxInfo> b_info) {
    auto overlap_1D = [](float x1_min, float x1_max, float x2_min, float x2_max) -> float {
        if (x1_min > x2_min) {
            std::swap(x1_min, x2_min);
            std::swap(x1_max, x2_max);
        }
        return x1_max < x2_min ? 0 : std::min(x1_max, x2_max) - x2_min;
    };

    auto compute_iou = [&overlap_1D](BBox& bbox1, BBox& bbox2) -> float {
        float overlap_x = overlap_1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlap_y = overlap_1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap_2D = overlap_x * overlap_y;
        float u = area1 + area2 - overlap_2D;
        return u == 0 ? 0 : overlap_2D / u;
    };

    std::stable_sort(b_info.begin(), b_info.end(), [](const BBoxInfo& b1, const BBoxInfo &b2) {
        return b1.prob > b2.prob;
    });

    std::vector<BBoxInfo> out;
#if defined(WEIGHTED_NMS)
    std::vector<std::vector < BBoxInfo> > weigthed_nms_candidates;
#endif
    for (auto& i : b_info) {
        bool keep = true;
#if defined(WEIGHTED_NMS)
        int j_index = 0;
#endif
        for (auto& j : out) {
            if (keep) {
                float overlap = compute_iou(i.box, j.box);
                keep = overlap <= nms_thresh;
#if defined(WEIGHTED_NMS)
                if (!keep && fabs(j.prob - i.prob) < 0.52f) // add label similarity check
                    weigthed_nms_candidates[j_index].push_back(i);
#endif
            } else
                break;
#if defined(WEIGHTED_NMS)  
            j_index++;
#endif
        }
        if (keep) {
            out.push_back(i);
#if defined(WEIGHTED_NMS)
            weigthed_nms_candidates.emplace_back();
            weigthed_nms_candidates.back().clear();
#endif
        }
    }
#if defined(WEIGHTED_NMS)
    for (int i = 0; i < out.size(); i++) {
        // the best confidence
        BBoxInfo& best = out[i];
        float sum_tl_x = best.box.x1 * best.prob;
        float sum_tl_y = best.box.y1 * best.prob;
        float sum_br_x = best.box.x2 * best.prob;
        float sum_br_y = best.box.y2 * best.prob;

        float weight = best.prob;
        for (auto& it : weigthed_nms_candidates[i]) {
            sum_tl_x += it.box.x1 * it.prob;
            sum_tl_y += it.box.y1 * it.prob;
            sum_br_x += it.box.x2 * it.prob;
            sum_br_y += it.box.y2 * it.prob;
            weight += it.prob;
        }

        weight = 1.f / weight;
        best.box.x1 = sum_tl_x * weight;
        best.box.y1 = sum_tl_y * weight;
        best.box.x2 = sum_br_x * weight;
        best.box.y2 = sum_br_y * weight;
    }
#endif
    return out;
}

inline bool readFile(std::string filename, std::vector<uint8_t> &file_content) {
    // open the file:
    std::ifstream instream(filename, std::ios::in | std::ios::binary);
    if (!instream.is_open()) return true;
    file_content = std::vector<uint8_t>((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());
    return false;
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