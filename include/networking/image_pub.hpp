//
// Created by prometheus on 1/20/23.
//

#ifndef VISION_SYSTEM_IMAGE_PUB_HPP
#define VISION_SYSTEM_IMAGE_PUB_HPP

#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"
#include "zmq_publisher.hpp"

struct image_publishable : public publishable {
    const std::string topic_ = "image";
    std::unique_ptr<uint8_t[]> data_ptr_;
    cv::Mat img_;
    uint8_t* get_byte_array() override {
        data_ptr_ = std::make_unique<uint8_t[]>(img_.total() * img_.elemSize());
        encode(data_ptr_.get());
        return data_ptr_.get();
    }

    void encode(uint8_t* buffer) override {
        encode_cv_mat(buffer, 0, img_.total() * img_.elemSize(), img_);
    }

    std::string get_topic() const override {
        return topic_;
    }

    size_t get_size() const override {
        return img_.total() * img_.elemSize();
    }
};

#endif //VISION_SYSTEM_IMAGE_PUB_HPP
