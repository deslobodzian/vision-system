#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <zmq.hpp>
#include <string>
#include <flatbuffers/flatbuffers.h>
#include "utils/logger.hpp"

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint)
        : publisher_(ZmqPublisher::context_, ZMQ_PUB) {
            publisher_.bind(endpoint);
            publisher_.set(zmq::sockopt::sndhwm, 1000); // Adjust the value based on your requirements
            LOG_DEBUG("Binded to endpoint: ", endpoint);
        }

    template<typename Func, typename... Args>
    void publish(const std::string& topic, Func&& create_func, Args&&... args) {
        std::lock_guard<std::mutex> lock(mtx_);
        builder_.Clear();
        auto offset = std::invoke(
            std::forward<Func>(create_func),
            builder_,
            std::forward<Args>(args)...
        );
        builder_.Finish(offset);
        send_message(topic, builder_.GetBufferPointer(), builder_.GetSize());
     }

    void publish_prebuilt(const std::string& topic, const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(mtx_);
        send_message(topic, data, size);
    }

    flatbuffers::FlatBufferBuilder& get_builder() {
        return builder_;
    }

private:
    static inline zmq::context_t context_{1};
    zmq::socket_t publisher_;
    flatbuffers::FlatBufferBuilder builder_;
    std::mutex mtx_;

    void send_message(const std::string& topic, const uint8_t* data, size_t size) {
        zmq::message_t topic_msg(topic.data(), topic.size());
        zmq::message_t data_msg(data, size);
        publisher_.send(topic_msg, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
        publisher_.send(data_msg, zmq::send_flags::dontwait);
    }
};


#endif /* VISION_SYSTEM_ZMQ_PUBLISHER_HPP */
