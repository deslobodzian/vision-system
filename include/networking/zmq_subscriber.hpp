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
        // This method now returns an optional pair containing the received topic and message data.
    std::optional<std::pair<std::string, zmq::message_t>> receive() {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;

        if (!subscriber_.recv(topic_msg, zmq::recv_flags::none)) {
            return std::nullopt;
        }

        if (!subscriber_.recv(data_msg, zmq::recv_flags::none)) {
            return std::nullopt;
        }

        // Convert the topic message to string for returning.
        std::string topic_received(static_cast<char*>(topic_msg.data()), topic_msg.size());

        return std::make_optional(std::make_pair(std::move(topic_received), std::move(data_msg)));
    }




private:
    zmq::context_t context_;
    zmq::socket_t subscriber_;
    std::string topic_;
};

#endif /* VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP */

