//
// Created by ubuntuvm on 11/9/22.
//

#ifndef VISION_SYSTEM_APRILTAG_DETECTOR_HPP
#define VISION_SYSTEM_APRILTAG_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include "vision/camera_config.hpp"
#include "utils/utils.hpp"
#include "tracked_target_info.hpp"

extern "C" {
    #include "apriltag.h"
    #include "tag36h11.h"
    #include "tag25h9.h"
    #include "tag16h5.h"
    #include "tagCircle21h7.h"
    #include "tagCircle49h12.h"
    #include "tagCustom48h12.h"
    #include "tagStandard41h12.h"
    #include "tagStandard52h13.h"
    #include "common/getopt.h"
    #include "apriltag_pose.h"
}

struct Corners {
    cv::Point tr;
    cv::Point tl;
    cv::Point bl;
    cv::Point br;
};

enum tag_family{
    tag36h11,
    tag25h9,
    tag16h5,
    tagCircle21h7,
    tagCircle49h12,
    tagStandard41h12,
    tagStandard52h13,
    tagCustom48h12
};

struct detector_config {
    tag_family tf;
    float quad_decimate;
    float quad_sigma;
    int nthreads;
    bool debug;
    bool refine_edges;
};

class TagDetector {
private:
    apriltag_detector *td_;
    apriltag_family_t *tf_ = NULL;

    apriltag_family_t* create_tag(tag_family tf);
    zarray_t* current_detections_;
    void destroy_tag(tag_family tag_family_class, apriltag_family_t *tf);
public:
    TagDetector();
    TagDetector(detector_config cfg);
    ~TagDetector();

    // preprocessed image that is gray, look into just adding raw image from monocular or zed camera
    zarray_t* get_detections(const cv::Mat &img);
    void fetch_detections(const cv::Mat &img);
    zarray_t* get_current_detections();
    int get_detections_size(const zarray_t *detections);
    int get_current_number_of_targets();
    bool has_targets();
    apriltag_detection_t* get_target_from_id(int id);

    cv::Point get_detection_center(apriltag_detection_t *det);
    Corners get_detection_corners(apriltag_detection_t *det);
    template <typename T>
    apriltag_pose_t get_estimated_target_pose(
            IntrinsicParameters<T> params,
            apriltag_detection_t *det,
            T tag_size
            );
    template <typename T>
    sl::Pose get_estimated_target_pose(const sl::Vector3<T> &tr, const sl::Vector3<T> &tl, const sl::Vector3<T> &br);
};

#endif //VISION_SYSTEM_APRILTAG_DETECTOR_HPP
