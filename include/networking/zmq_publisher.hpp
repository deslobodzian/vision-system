#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include "utils/logger.hpp"
#include <flatbuffers/flatbuffers.h>
#include <string>
#include <zmq.hpp>

class ZmqPublisher {
 public:
  explicit ZmqPublisher(const std::string& endpoint)
      : context_(1), publisher_(context_, ZMQ_PUB) {
    publisher_.bind(endpoint);
    publisher_.set(zmq::sockopt::sndhwm, 1000);
    LOG_DEBUG("Binded to endpoint: ", endpoint);
  }

  template <typename Func, typename... Args>
  void publish(const std::string& topic, Func&& create_func, Args&&... args) {
    std::lock_guard<std::mutex> lock(mtx_);
    builder_.Clear();
    auto offset = std::invoke(std::forward<Func>(create_func), builder_,
                              std::forward<Args>(args)...);
    builder_.Finish(offset);
    send_message(topic, builder_.GetBufferPointer(), builder_.GetSize());
  }

  void publish_prebuilt(const std::string& topic, const uint8_t* data,
                        size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    send_message(topic, data, size);
    builder_.Clear();
  }

  flatbuffers::FlatBufferBuilder& get_builder() { return builder_; }

 private:
  zmq::context_t context_;
  zmq::socket_t publisher_;
  flatbuffers::FlatBufferBuilder builder_;
  std::mutex mtx_;

  void send_message(const std::string& topic, const uint8_t* data,
                    size_t size) {
    zmq::message_t topic_msg(topic.data(), topic.size());
    zmq::message_t data_msg(data, size);
    publisher_.send(topic_msg, zmq::send_flags::sndmore);
    publisher_.send(data_msg, zmq::send_flags::dontwait);
  }
};

#endif /* VISION_SYSTEM_ZMQ_PUBLISHER_HPP */
