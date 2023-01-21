//
// Created by robot on 1/16/23.
//

#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <string>
#include <iostream>
#include <zmq.h>

struct publishable {
    virtual uint8_t* get_byte_array() {return nullptr;}
    virtual std::string get_topic() const = 0;
    virtual void encode(uint8_t* buffer) = 0;
    virtual size_t get_size() const = 0;
//    std::mutex publishable_mtx_;
};

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint, publishable* p) {
        context_ = zmq_ctx_new();
        publisher_ = zmq_socket(context_, ZMQ_PUB);
        info("[Publisher][" + p->get_topic() + "]: Binding socket to: " + endpoint);
        int rc = zmq_bind(publisher_, endpoint.c_str());
        publishable_ = p;
    }

    ~ZmqPublisher() {
        zmq_close(publisher_);
        zmq_ctx_destroy(context_);
    }

    void send() {
        std::string message = "Test";
        zmq_send(publisher_, message.c_str(), message.size(), ZMQ_SNDMORE);
        std::string message_1 = "Works";
        zmq_send(publisher_, message_1.c_str(), message_1.size(), 0);

//        publisher_->send(publishable_->get_byte_array(), publishable_->get_size());
    }

private:
    void* context_;
    void* publisher_;
    publishable* publishable_;
};

#endif //VISION_SYSTEM_ZMQ_PUBLISHER_HPP
