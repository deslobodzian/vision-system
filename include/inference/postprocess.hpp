#ifndef VISION_SYSTEM_POSTPROCESS_HPP
#define VISION_SYSTEM_POSTPROCESS_HPP

#include "bbox.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

inline int clamp(int val, int min, int max) {
  if (val <= min)
    return min;
  if (val >= max)
    return max;
  return val;
}

#define WEIGHTED_NMS

inline std::vector<BBoxInfo>
non_maximum_suppression(const float nms_thresh, std::vector<BBoxInfo> b_info) {
  auto overlap_1D = [](float x1_min, float x1_max, float x2_min,
                       float x2_max) -> float {
    if (x1_min > x2_min) {
      std::swap(x1_min, x2_min);
      std::swap(x1_max, x2_max);
    }
    return x1_max < x2_min ? 0 : std::min(x1_max, x2_max) - x2_min;
  };

  auto compute_iou = [&overlap_1D](BBox &bbox1, BBox &bbox2) -> float {
    float overlap_x = overlap_1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
    float overlap_y = overlap_1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
    float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
    float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
    float overlap_2D = overlap_x * overlap_y;
    float u = area1 + area2 - overlap_2D;
    return u == 0 ? 0 : overlap_2D / u;
  };

  std::stable_sort(b_info.begin(), b_info.end(),
                   [](const BBoxInfo &b1, const BBoxInfo &b2) {
                     return b1.probability > b2.probability;
                   });

  std::vector<BBoxInfo> out;
#if defined(WEIGHTED_NMS)
  std::vector<std::vector<BBoxInfo>> weigthed_nms_candidates;
#endif
  for (auto &i : b_info) {
    bool keep = true;
#if defined(WEIGHTED_NMS)
    int j_index = 0;
#endif
    for (auto &j : out) {
      if (keep) {
        float overlap = compute_iou(i.box, j.box);
        keep = overlap <= nms_thresh;
#if defined(WEIGHTED_NMS)
        if (!keep && fabs(j.probability - i.probability) <
                         0.52f) // add label similarity check
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
    BBoxInfo &best = out[i];
    float sum_tl_x = best.box.x1 * best.probability;
    float sum_tl_y = best.box.y1 * best.probability;
    float sum_br_x = best.box.x2 * best.probability;
    float sum_br_y = best.box.y2 * best.probability;

    float weight = best.probability;
    for (auto &it : weigthed_nms_candidates[i]) {
      sum_tl_x += it.box.x1 * it.probability;
      sum_tl_y += it.box.y1 * it.probability;
      sum_br_x += it.box.x2 * it.probability;
      sum_br_y += it.box.y2 * it.probability;
      weight += it.probability;
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

#endif /* VISION_SYSTEM_POSTPROCESS_HPP */
