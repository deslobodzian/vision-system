//
// Created by deslobodzian on 11/9/22.
//

#include "vision/apriltag_detector.hpp"
#include "utils.hpp"

TagDetector::TagDetector() {
    tf_ = tag36h11_create();
    td_ = apriltag_detector_create();

    apriltag_detector_add_family(td_, tf_);
}

TagDetector::TagDetector(DetectorConfig cfg) {
    tf_ = create_tag(cfg.tf);
    td_ = apriltag_detector_create();

    apriltag_detector_add_family(td_, tf_);

    td_->quad_decimate = cfg.quad_decimate;
    td_->quad_sigma = cfg.quad_sigma;
    td_->nthreads = cfg.nthreads;
    td_->debug = cfg.debug;
    td_->refine_edges = cfg.refine_edges;
}

TagDetector::~TagDetector() {}

zarray_t* TagDetector::get_detections(cv::Mat img) {
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
void TagDetector::fetch_detections(cv::Mat img) {
    current_detections_ = get_detections(img);
    if(errno == EAGAIN) {
	    info("unable to create threads");
    }
}

int TagDetector::get_detections_size(const zarray_t *detections) {
    return zarray_size(detections);
}

int TagDetector::get_current_number_of_targets() {
    return get_detections_size(current_detections_);
}

cv::Point TagDetector::get_detection_center(apriltag_detection_t *det) {
    return cv::Point(det->c[0], det->c[1]);
}
Corners TagDetector::get_detection_corners(apriltag_detection_t *det) {
    cv::Point tr(det->p[0][0], det->p[0][1]);
    cv::Point tl(det->p[1][0], det->p[1][1]);
    cv::Point bl(det->p[2][0], det->p[2][1]);
    cv::Point br(det->p[3][1], det->p[3][1]);
    return {tr, tl, bl, br};
}

apriltag_family_t* TagDetector::create_tag(tag_family tf) {
    switch(tf) {
        case tag36h11:
            return tag36h11_create();
        case tag25h9:
            return tag25h9_create();
        case tag16h6:
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

void TagDetector::destroy_tag(tag_family tag_family_class, apriltag_family_t *tf) {
    switch(tag_family_class) {
        case tag36h11:
            tag36h11_destroy(tf);
            break;
        case tag25h9:
            tag25h9_destroy(tf);
            break;
        case tag16h6:
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
