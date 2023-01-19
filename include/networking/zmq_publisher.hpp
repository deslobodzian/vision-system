//
// Created by robot on 1/16/23.
//

#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <string>
#include <iostream>
#include <zmq.hpp>

struct publishable {
    virtual uint8_t* get_byte_array() {return nullptr;}
    virtual std::string get_topic() const = 0;
    virtual void encode(uint8_t* buffer) = 0;
    virtual size_t get_size() = 0;
//    std::mutex publishable_mtx_;
};

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint) {
        publisher_ = new zmq::socket_t(context_, ZMQ_PUB);
        publisher_->bind(endpoint);
    }

    ~ZmqPublisher() {
        publisher_->close();
        delete publisher_;
    }

    template <typename T>
    void send(const T& publishable) {
        publisher_->send(publishable);
        publisher_->send(publishable.get_byte_array(), publishable.get_size());
    }

private:
    zmq::context_t context_ = zmq::context_t(1);
    zmq::socket_t* publisher_;
};

#endif //VISION_SYSTEM_ZMQ_PUBLISHER_HPP
