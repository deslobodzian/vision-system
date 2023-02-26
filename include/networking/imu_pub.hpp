//
// Created by robot on 2/26/23.
//

#ifndef VISION_SYSTEM_IMU_PUB_HPP
#define VISION_SYSTEM_IMU_PUB_HPP
#include "zmq_publisher.hpp"
#include <memory>
#include "utils/utils.hpp"
#include "vision/tracked_target_info.hpp"

struct imu_publishable : public publishable {
    const std::string topic_ = "imu";
    float imu_data_[3]; // (yaw, pitch, roll)
    std::unique_ptr<uint8_t[]> data_ptr_;

    uint8_t* get_byte_array() override {
        data_ptr_ = std::make_unique<uint8_t[]>(get_size());
        encode(data_ptr_.get());
        return data_ptr_.get();
    }

    void encode(uint8_t *buffer) override {
        encode_float_array(buffer, 0, get_size(), imu_data_, 3);
    }

    size_t get_size() const override {
        return (3 * sizeof(float));
    }

    std::string get_topic() const override {
        return topic_;
    }
};
#endif //VISION_SYSTEM_IMU_PUB_HPP
