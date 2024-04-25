#ifndef VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP
#define VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP

#include "utils/logger.hpp"
#include <flatbuffers/flatbuffers.h>
#include <optional>
#include <string>
#include <utility>
#include <zmq.hpp>

class ZmqSubscriber {
 public:
  ZmqSubscriber(std::string topic, const std::string& endpoint)
      : topic_(std::move(topic)),
        subscriber_(ZmqSubscriber::context_, ZMQ_SUB) {
    subscriber_.connect(endpoint);
    subscriber_.set(zmq::sockopt::subscribe, topic_);
    subscriber_.set(zmq::sockopt::rcvhwm, 100);
    LOG_DEBUG("Connected to endpoint: ", endpoint, " with topic: ", topic_);
  }

  std::optional<std::pair<std::string, zmq::message_t>> receive() {
    std::lock_guard<std::mutex> lock(mtx_);
    zmq::message_t topic_msg;
    zmq::message_t data_msg;

    if (!subscriber_.recv(topic_msg, zmq::recv_flags::dontwait)) {
      return std::nullopt;
    }

    if (!subscriber_.recv(data_msg, zmq::recv_flags::dontwait)) {
      return std::nullopt;
    }

    std::string topic_received(static_cast<char*>(topic_msg.data()),
                               topic_msg.size());

    return std::make_optional(
        std::make_pair(std::move(topic_received), std::move(data_msg)));
  }

 private:
  static inline zmq::context_t context_{1};
  zmq::socket_t subscriber_;
  std::string topic_;
  std::mutex mtx_;
};

#endif /* VISION_SYSTEM_ZMQ_SUBSCRIBER_HPP */
