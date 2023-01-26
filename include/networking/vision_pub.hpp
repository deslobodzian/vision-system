//
// Created by robot on 1/17/23.
//

#ifndef VISION_SYSTEM_VISION_PUB_HPP
#define VISION_SYSTEM_VISION_PUB_HPP

#include "zmq_publisher.hpp"
#include <memory>
#include "utils/utils.hpp"
#include "vision/tracked_target_info.hpp"

struct vision_publishable : public publishable {
    const std::string topic_ = "vision";
    std::vector<tracked_target_info> targets_;
    std::unique_ptr<uint8_t[]> data_ptr_;

    uint8_t* get_byte_array() override {
        data_ptr_ = std::make_unique<uint8_t[]>(get_size());
        encode(data_ptr_.get());
        return data_ptr_.get();
    }

    void encode(uint8_t *buffer) override {
        int offset = 0;
        for (auto &target : targets_) {
            target.encode(buffer + offset);
            offset += tracked_target_info::size();
        }
    }

    size_t get_size() const override {
        return (7 * sizeof(float)) * targets_.size();
    }

    std::string get_topic() const override {
        return topic_;
    }
};
#endif //VISION_SYSTEM_VISION_PUB_HPP
