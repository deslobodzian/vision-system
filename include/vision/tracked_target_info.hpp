//
// Created by DSlobodzian on 4/21/2022.
//

#ifndef VISION_SYSTEM_TARGET_INFO_HPP
#define VISION_SYSTEM_TARGET_INFO_HPP

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include "apriltag_detector.hpp"
#include <Eigen/Dense>
#include "map.hpp"

struct tracked_target_info {
    uint8_t id_;
    float x_;
    float y_;
    float z_;
    float vx_;
    float vy_;
    float vz_;

    explicit tracked_target_info(const sl::ObjectData& data) :
        id_(data.raw_label),
        x_(data.position.x),
        y_(data.position.y),
        z_(data.position.z),
        vx_(data.velocity.x),
        vy_(data.velocity.y),
        vz_(data.velocity.z) {
    }

    explicit tracked_target_info(const float x, const float y, const float z, const uint8_t id) :
        id_(id),
        x_(x),
        y_(y),
        z_(z),
        vx_(0.0f),
        vy_(0.0f),
        vz_(0.0f) {
    }

    void encode(uint8_t* buffer) {
        memcpy(buffer, &id_, sizeof(uint8_t));
        memcpy(buffer + sizeof(uint8_t), &x_, sizeof(float) * 6);
    }

    static int size() {
        return sizeof(uint8_t) + (sizeof(float) * 6);
    }
};

#endif //VISION_SYSTEM_TARGET_INFO_HPP
