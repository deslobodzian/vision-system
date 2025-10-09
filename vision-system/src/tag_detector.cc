// tag_detector.cpp
#include "tag_detector.h"
#include "apriltag.h"
#include "tag36h11.h"

TagDetector::TagDetector() 
    : detector_(apriltag_detector_create()),
      tag_family_(tag36h11_create()),
      detections_(nullptr) {
    
    if (detector_ && tag_family_) {
        apriltag_detector_add_family(detector_, tag_family_);
    }
}

TagDetector::~TagDetector() {
    if (detections_) {
        apriltag_detections_destroy(detections_);
    }
    if (detector_ && tag_family_) {
        apriltag_detector_remove_family(detector_, tag_family_);
    }
    if (tag_family_) {
        tag36h11_destroy(tag_family_);
    }
    if (detector_) {
        apriltag_detector_destroy(detector_);
    }
}

TagDetector::TagDetector(const TagDetector& other) 
    : detector_(apriltag_detector_create()),
      tag_family_(tag36h11_create()),
      detections_(nullptr) {
    
    if (detector_ && tag_family_) {
        apriltag_detector_add_family(detector_, tag_family_);
    }
    
    if (other.detector_) {
        copy_settings(other.detector_);
    }
}

TagDetector& TagDetector::operator=(const TagDetector& other) {
    if (this != &other) {
        if (detections_) {
            apriltag_detections_destroy(detections_);
            detections_ = nullptr;
        }
        
        if (other.detector_) {
            copy_settings(other.detector_);
        }
    }
    return *this;
}

TagDetector::TagDetector(TagDetector&& other) noexcept 
    : detector_(other.detector_),
      tag_family_(other.tag_family_),
      detections_(other.detections_) {
    
    other.detector_ = nullptr;
    other.tag_family_ = nullptr;
    other.detections_ = nullptr;
}

TagDetector& TagDetector::operator=(TagDetector&& other) noexcept {
    if (this != &other) {
        if (detections_) {
            apriltag_detections_destroy(detections_);
        }
        if (detector_ && tag_family_) {
            apriltag_detector_remove_family(detector_, tag_family_);
        }
        if (tag_family_) {
            tag36h11_destroy(tag_family_);
        }
        if (detector_) {
            apriltag_detector_destroy(detector_);
        }
        
        detector_ = other.detector_;
        tag_family_ = other.tag_family_;
        detections_ = other.detections_;
        
        other.detector_ = nullptr;
        other.tag_family_ = nullptr;
        other.detections_ = nullptr;
    }
    return *this;
}

image_u8_t* TagDetector::mat_to_image_u8(const cv::Mat& mat) {
    cv::Mat gray;
    
    if (mat.channels() == 3) {
        cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = mat;
    }
    
    if (!gray.isContinuous()) {
        gray = gray.clone();
    }
    
    image_u8_t* im = image_u8_create(gray.cols, gray.rows);
    memcpy(im->buf, gray.data, gray.cols * gray.rows);
    
    return im;
}

std::vector<TagDetection> TagDetector::detect(const cv::Mat& image) {
    std::vector<TagDetection> results;
    
    if (!detector_ || image.empty()) {
        return results;
    }
    
    clear_detections();
    
    image_u8_t* im = mat_to_image_u8(image);
    
    detections_ = apriltag_detector_detect(detector_, im);
    
    for (int i = 0; i < zarray_size(detections_); i++) {
        apriltag_detection_t* det;
        zarray_get(detections_, i, &det);
        
        TagDetection tag;
        tag.id = det->id;
        tag.center[0] = det->c[0];
        tag.center[1] = det->c[1];
        tag.decision_margin = det->decision_margin;
        tag.hamming = det->hamming;
        
        for (int j = 0; j < 4; j++) {
            tag.corners[j][0] = det->p[j][0];
            tag.corners[j][1] = det->p[j][1];
        }
        
        results.push_back(tag);
    }
    
    image_u8_destroy(im);
    
    return results;
}

zarray_t* TagDetector::detect_raw(const cv::Mat& image) {
    if (!detector_ || image.empty()) {
        return nullptr;
    }
    
    clear_detections();
    
    image_u8_t* im = mat_to_image_u8(image);
    
    detections_ = apriltag_detector_detect(detector_, im);
    
    image_u8_destroy(im);
    
    return detections_;
}

void TagDetector::clear_detections() {
    if (detections_) {
        apriltag_detections_destroy(detections_);
        detections_ = nullptr;
    }
}

void TagDetector::copy_settings(const apriltag_detector_t* src) {
    if (!detector_ || !src) return;
    
    detector_->nthreads = src->nthreads;
    detector_->quad_decimate = src->quad_decimate;
    detector_->quad_sigma = src->quad_sigma;
    detector_->refine_edges = src->refine_edges;
    detector_->decode_sharpening = src->decode_sharpening;
}