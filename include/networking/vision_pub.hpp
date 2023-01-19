//
// Created by robot on 1/17/23.
//

#ifndef VISION_SYSTEM_VISION_PUB_HPP
#define VISION_SYSTEM_VISION_PUB_HPP

#include "zmq_publisher.hpp"
#include <memory>
#include "utils/utils.hpp"

struct vision_publishable : public publishable {
    const std::string topic_ = "vision";
    float target_info[3];
    std::unique_ptr<uint8_t[]> data_ptr_;
    size_t size_ = 3 * sizeof(float);

    uint8_t* get_byte_array() override {
        data_ptr_ = std::make_unique<uint8_t[]>(size_);
        encode(data_ptr_.get());
        return data_ptr_.get();
    }

    void encode(uint8_t *buffer) override {
        encode_float_array(buffer, 0, size_, target_info, 3);
    }

    size_t get_size() {
        return size_;
    }

    std::string get_topic() const override {
        return topic_;
    }
};
#endif //VISION_SYSTEM_VISION_PUB_HPP
