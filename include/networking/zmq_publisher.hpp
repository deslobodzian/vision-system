#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <zmq.hpp>
#include <string>
#include <flatbuffers/flatbuffers.h>
#include "utils/logger.hpp"

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint)
        : context_(1), publisher_(context_, ZMQ_PUB) {
            publisher_.bind(endpoint);
            LOG_DEBUG("Created publisher with endpoint: ", endpoint);
        }

    template<typename Func, typename... Args>
    void publish(const std::string& topic, Func&& create_func, Args&&... args) {
        std::lock_guard<std::mutex> lock(mtx_);
        builder_.Clear();
        // kind of ugly but its the closest thing to what I want. 
        // I really wish I could call publish(args...), TODO: Think harder
        auto offset = std::invoke(
                std::forward<Func>(create_func),
                builder_,
                std::forward<Args>(args)...
                );
        builder_.Finish(offset);
        zmq::message_t message(builder_.GetBufferPointer(), builder_.GetSize());
        publisher_.send(zmq::buffer(topic), zmq::send_flags::sndmore);
        publisher_.send(message, zmq::send_flags::none);
    }

    flatbuffers::FlatBufferBuilder& get_builder() {
        return builder_;
    }

private:
    zmq::context_t context_;
    zmq::socket_t publisher_;
    flatbuffers::FlatBufferBuilder builder_;
    std::mutex mtx_;
};

#endif /* VISION_SYSTEM_ZMQ_PUBLISHER_HPP */
