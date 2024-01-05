#ifndef VISION_SYSTEM_ZMQ_MANAGER_HPP
#define VISION_SYSTEM_ZMQ_MANAGER_HPP


#include <map>
#include "zmq_publisher.hpp"
#include <flatbuffers/flatbuffer_builder.h>

class ZmqManager {
public:
    ZmqManager() = default;

    ~ZmqManager() {
    }

private:
    flatbuffers::FlatBufferBuilder builder_;
};
#endif /* VISION_SYSTEM_ZMQ_MANAGER_HPP */
