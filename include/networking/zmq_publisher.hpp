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
        debug("[Publisher][" + p->get_topic() + "]: Binding socket to: " + endpoint);
        int rc = zmq_bind(publisher_, endpoint.c_str());
        publishable_ = p;
    }

    ~ZmqPublisher() {
        zmq_close(publisher_);
        zmq_ctx_destroy(context_);
    }

    void send() {
        zmq_send(publisher_, publishable_->get_topic().c_str(), publishable_->get_topic().size(), ZMQ_SNDMORE);
        zmq_send(publisher_, publishable_->get_byte_array(), publishable_->get_size(),0);
    }

private:
    void* context_;
    void* publisher_;
    publishable* publishable_;
};

#endif //VISION_SYSTEM_ZMQ_PUBLISHER_HPP
