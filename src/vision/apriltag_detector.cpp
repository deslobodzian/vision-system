//
// Created by deslobodzian on 11/9/22.
//

#include "vision/apriltag_detector.hpp"
#include "utils/utils.hpp"

template <typename T>
TagDetector<T>::TagDetector() {
    tf_ = tag36h11_create();
    td_ = apriltag_detector_create();

    apriltag_detector_add_family(td_, tf_);
}

template <typename T>
TagDetector<T>::TagDetector(detector_config cfg) {
    tf_ = create_tag(cfg.tf);
    td_ = apriltag_detector_create();

    apriltag_detector_add_family(td_, tf_);

    td_->quad_decimate = cfg.quad_decimate;
    td_->quad_sigma = cfg.quad_sigma;
    td_->nthreads = cfg.nthreads;
    td_->debug = cfg.debug;
    td_->refine_edges = cfg.refine_edges;
}

template <typename T>
TagDetector<T>::~TagDetector() {
    apriltag_detector_destroy(td_);
    apriltag_detections_destroy(current_detections_);
}

template <typename T>
zarray_t* TagDetector<T>::get_detections(const cv::Mat &img) {
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    image_u8_t im = {
            .width = gray_img.cols,
            .height = gray_img.rows,
            .stride = gray_img.cols,
            .buf = gray_img.data
    };
    return apriltag_detector_detect(td_, &im);
}

template <typename T>
void TagDetector<T>::fetch_detections(const cv::Mat &img) {
    current_detections_ = get_detections(img);
}

template <typename T>
zarray_t* TagDetector<T>::get_current_detections() {
    return current_detections_;
}

template <typename T>
int TagDetector<T>::get_detections_size(const zarray_t *detections) {
    return zarray_size(detections);
}

template <typename T>
int TagDetector<T>::get_current_number_of_targets() {
    return get_detections_size(current_detections_);
}

template <typename T>
bool TagDetector<T>::has_targets() {
    return get_current_number_of_targets() > 0;
}

template <typename T>
apriltag_detection_t* TagDetector<T>::get_target_from_id(int id) {
    apriltag_detection_t *det;
    for (int i = 0; i < get_current_number_of_targets(); i++) {
        zarray_get(current_detections_, i, &det);
        if (id == det->id) {
            return det;
        }
    }
    return nullptr;
}
template <typename T>
cv::Point TagDetector<T>::get_detection_center(apriltag_detection_t *det) {
    return {(int)det->c[0], (int)det->c[1]};
}

template <typename T>
Corners TagDetector<T>::get_detection_corners(apriltag_detection_t *det) {
    cv::Point tr((int)det->p[0][0], (int)det->p[0][1]);
    cv::Point tl((int)det->p[1][0], (int)det->p[1][1]);
    cv::Point bl((int)det->p[2][0], (int)det->p[2][1]);
    cv::Point br((int)det->p[3][1], (int)det->p[3][1]);
    return {tr, tl, bl, br};
}

template <typename T>
apriltag_pose_t TagDetector<T>::get_estimated_target_pose(
        IntrinsicParameters<T> params,
        apriltag_detection_t *det,
        T tag_size) {
    apriltag_detection_info_t info;
    info.det = det;
    info.tagsize = tag_size;
    info.fx = params.fx;
    info.fy = params.fy;
    info.cx = params.cx;
    info.cy = params.cy;

    apriltag_pose_t pose;
    T err = estimate_tag_pose(&info, &pose);
    return pose;
}

template <typename T>
sl::Pose TagDetector<T>::get_estimated_target_pose(const sl::Vector3<T> &tr, const sl::Vector3<T> &tl, const sl::Vector3<T> &br) {
    sl::Vector3<T> normal_vec = calculate_plane_normal_vector(tr, tl, br);
    sl::Orientation orientation(orientation_from_normal_vec(normal_vec));
    sl::Translation translation((tl + br) / 2.0);
    return {sl::Transform(orientation, translation)};
}

template <typename T>
apriltag_family_t* TagDetector<T>::create_tag(tag_family tf) {
    switch(tf) {
        case tag36h11:
            return tag36h11_create();
        case tag25h9:
            return tag25h9_create();
        case tag16h5:
            return tag16h5_create();
        case tagCircle21h7:
            return tagCircle21h7_create();
        case tagCircle49h12:
            return tagCircle49h12_create();
        case tagStandard41h12:
            return tagStandard41h12_create();
        case tagStandard52h13:
            return tagStandard52h13_create();
        case tagCustom48h12:
            return tagCustom48h12_create();
        default:
            return tag36h11_create(); // in case tf is not one of the enum values
    }
}

template <typename T>
void TagDetector<T>::destroy_tag(tag_family tag_family_class, apriltag_family_t *tf) {
    switch(tag_family_class) {
        case tag36h11:
            tag36h11_destroy(tf);
            break;
        case tag25h9:
            tag25h9_destroy(tf);
            break;
        case tag16h5:
            tag16h5_destroy(tf);
            break;
        case tagCircle21h7:
            tagCircle21h7_destroy(tf);
            break;
        case tagCircle49h12:
            tagCircle49h12_destroy(tf);
            break;
        case tagStandard41h12:
            tagStandard41h12_destroy(tf);
            break;
        case tagStandard52h13:
            tagStandard52h13_destroy(tf);
            break;
        case tagCustom48h12:
            tagCustom48h12_destroy(tf);
            break;
        default:
            tag36h11_destroy(tf); // in case tf is not one of the enum values
            break;
    }
}

template class TagDetector<float>;
template class TagDetector<double>;
