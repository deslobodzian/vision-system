#ifndef VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP
#define VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP

#include <zmq.hpp>
#include <string>
#include <flatbuffers/flatbuffers.h>
#include "utils/logger.hpp"

class ZmqSubscriber {
public:
    ZmqSubscriber(const std::string& topic, const std::string& endpoint)
        : topic_(topic), context_(1), subscriber_(context_, ZMQ_SUB) {
            LOG_DEBUG("Subscriber created with topic: ", topic_);
            subscriber_.connect(endpoint);
            subscriber_.set(zmq::sockopt::subscribe, topic_);
        }

    template<typename T>
    std::optional<T> receive() {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;

        if (!subscriber_.recv(topic_msg, zmq::recv_flags::none)) {
            return std::nullopt;
        }

        if (!subscriber_.recv(data_msg, zmq::recv_flags::none)) {
            return std::nullopt;
        }

        auto verifier = flatbuffers::Verifier(static_cast<uint8_t*>(data_msg.data()), data_msg.size());
        if (!T::Verify(verifier)) {
            LOG_ERROR("Invalid message received");
            return std::nullopt;
        }

        return T::Get(data_msg.data());
    }

private:
    zmq::context_t context_;
    zmq::socket_t subscriber_;
    std::string topic_;
};

#endif /* VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP */

