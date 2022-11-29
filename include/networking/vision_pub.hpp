//
// Created by ubuntuvm on 11/23/22.
//

#ifndef VISION_SYSTEM_VISION_PUB_HPP
#define VISION_SYSTEM_VISION_PUB_HPP

#include "nt_publisher.hpp"

struct vision_publishable : public publishable {
    const std::string topic_ = "vision";
    float target_info[3];

    std::span<uint8_t> to_span() override {
        uint8_t bytes[3 * sizeof(float)];
        encode(bytes);
        return {bytes};
    }

    void encode(uint8_t *buffer) override {
        encode_float_array(buffer, 0, sizeof(float) * 3, target_info, 3);
    }

    std::string get_topic() const override {
        return topic_;
    }
};

#endif //VISION_SYSTEM_VISION_PUB_HPP
