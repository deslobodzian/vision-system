#ifndef VISION_SYSTEM_ZMQ_FLATBUFFERS_UTILS
#define VISION_SYSTEM_ZMQ_FLATBUFFERS_UTILS

#include <zmq.hpp>
#include <flatbuffers/flatbuffers.h>
#include "use_detection_generated.h"
#include "logger.hpp"

inline bool process_use_detection(const zmq::message_t& msg) {
    auto verifier = flatbuffers::Verifier(static_cast<const uint8_t*>(msg.data()), msg.size());
    if (!Messages::VerifyUseDetectionBuffer(verifier)) {
        LOG_ERROR("Invalid UseDetection message received");
        return false;
    }
    const auto* use_detection = Messages::GetUseDetection(msg.data());
    if (!use_detection) {
        LOG_ERROR("Failed to get UseDetection data");
        return false;
    }

    return use_detection->use_detection();
}

#endif /* VISION_SYSTEM_ZMQ_FLATBUFFERS_UTILS */
