//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP
#define VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP

#include "nt_publisher.hpp"

struct state_estimate_publishable : public publishable {
    const std::string topic_ = "state_estimate";
    double state_estimate[3];

    std::span<uint8_t> to_span() {
        uint8_t bytes[3 * sizeof(double)];
        encode(bytes);
        return {bytes};
    }

    void encode(uint8_t *buffer) {
        memcpy(&state_estimate, buffer, sizeof(state_estimate));
    }

    std::string get_topic() const {
        return topic_;
    }
};

#endif //VISION_SYSTEM_STATE_ESTIMATE_PUB_HPP
