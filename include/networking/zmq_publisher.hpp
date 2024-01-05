#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <zmq.hpp>
#include <string>
#include <flatbuffers/flatbuffers.h>
#include "utils/logger.hpp"

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint, const std::string& topic)
        : topic_(topic), context_(1), publisher_(context_, ZMQ_PUB) {
        LOG_DEBUG(topic_);
        publisher_.bind(endpoint);
    }

    template<typename Func, typename... Args>
    void publish(Func&& create_func, Args&&... args) {
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
        publisher_.send(zmq::buffer(topic_), zmq::send_flags::sndmore);
        publisher_.send(message, zmq::send_flags::none);
    }
private:
    zmq::context_t context_;
    zmq::socket_t publisher_;
    flatbuffers::FlatBufferBuilder builder_;
    std::string topic_;
};



#endif /* VISION_SYSTEM_ZMQ_PUBLISHER_HPP */
