//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP
#define VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP

#include "nt_publisher.hpp"
#include "utils/utils.hpp"

struct state_estimate_publishable : public publishable {
    const std::string topic_ = "state_estimate";
    double state_estimate[3];

    std::span<uint8_t> to_span() override {
        uint8_t bytes[3 * sizeof(double)];
        encode(bytes);
        return {bytes};
    }

    void encode(uint8_t *buffer) override {
        encode_double_array(buffer, 0, sizeof(double) * 3, state_estimate, 3);
    }

    std::string get_topic() const override {
        return topic_;
    }
};

#endif //VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP
