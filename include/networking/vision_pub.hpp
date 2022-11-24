//
// Created by ubuntuvm on 11/23/22.
//

#ifndef VISION_SYSTEM_VISION_PUB_HPP
#define VISION_SYSTEM_VISION_PUB_HPP

#include "nt_publisher.hpp"

struct vision_publishable : public Publishable {
    double target_info[3];
    double pose_estimate[3];

    std::span<double> to_span() {
        return {target_info, pose_estimate};
    }
    std::string get_topic() const {
        return topic_;
    }
    const std::string topic_ = "vision";
};

#endif //VISION_SYSTEM_VISION_PUB_HPP
